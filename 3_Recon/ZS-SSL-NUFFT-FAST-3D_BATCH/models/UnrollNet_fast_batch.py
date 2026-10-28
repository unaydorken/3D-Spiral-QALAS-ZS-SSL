import torch
import torch.nn as nn
import numpy as np
from copy import copy

from torch.utils.checkpoint import checkpoint

from models.ResNet import ResNet
import models.data_consistency_fast_batch as ssdu_dc
import cupy as cp


lamda_start = 0.1


class UnrolledNet(nn.Module):

    def __init__(
        self,
        args,
        operators,
        toeplitz_operators=None,
        toeplitz_kernels=None,
        device=torch.device("cuda:0"),
        ndim=2,
    ):

        super().__init__()

        self.args = args
        self.device = device
        self.ndim = ndim

        # CG tolerances for each unroll block
        if getattr(args, 'progressive_cg', False):

            if args.nb_unroll_blocks == 1:
                cg_tols = [args.CG_tol]
            else:
                cg_tols = np.geomspace(
                    1e-1,
                    args.CG_tol,
                    args.nb_unroll_blocks
                )

        else:
            cg_tols = [args.CG_tol] * args.nb_unroll_blocks

        self.cg_args = []

        for tol in cg_tols:
            block_args = copy(args)
            block_args.CG_tol = float(tol)
            block_args.cg_stats = {'forward': [], 'backward': []}
            self.cg_args.append(block_args)

        print('CG tolerances:', [f'{t:.2e}' for t in cg_tols])

        if self.ndim not in (2, 3):
            raise ValueError(f"Unsupported ndim: {self.ndim}")

        self.operators          = operators
        self.toeplitz_operators = toeplitz_operators
        self.toeplitz_kernels   = toeplitz_kernels
        self.ncontrast          = len(operators)

        self.regularizer = ResNet(
            self.device,
            in_ch=2 * self.ncontrast,
            num_of_resblocks=args.nb_res_blocks,
            ndim=self.ndim,
            features=getattr(args, 'nb_features', 8),
        )

        self.lam = nn.Parameter(
            torch.tensor(
                [getattr(args, 'lambda_start', 0.1)],
                dtype=torch.float32,
            )
        )

        # GPU kernel cache: {kernel_index -> stacked CuPy array}
        # Shape of each cached value: [contrast, *2spatial].
        # Populated lazily and cleared after each optimizer step.
        self._kernel_gpu_cache: dict = {}

        # Persistent scratch workspace for the batched-contrast SENSE Toeplitz
        # operator.  It is deliberately NOT cleared after every optimizer step;
        # reusing this allocation is important for 3D speed.
        self._fast_dc_workspace = None


    def clear_kernel_cache(self):
        """Free mask-specific GPU Toeplitz kernels.

        The persistent SENSE maps / FFT scratch workspace is retained so the
        next unroll/epoch does not pay the large 3D allocation cost again.
        """
        self._kernel_gpu_cache.clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def report_cg_diagnostics(self):
        print('\nCG diagnostics:')

        for i, block_args in enumerate(self.cg_args):
            for phase in ('forward', 'backward'):
                values = block_args.cg_stats[phase]

                if not values:
                    continue

                iterations = np.array([v[0] for v in values])
                residuals = np.array([v[1] for v in values])

                print(
                    f'UB{i+1} {phase}: '
                    f'tol={block_args.CG_tol:.1e}, '
                    f'CG avg={iterations.mean():.1f}, '
                    f'CG max={iterations.max()}, '
                    f'max residual={residuals.max():.2e}'
                )

                values.clear()


    def _get_active_kernels(self, kernel_indices):
        """Upload and cache only the kernel sets needed by this batch.

        Kernels are stored as ONE stacked CuPy array per SSDU mask set:

            [ncontrast, 2*Nx, 2*Ny, 2*Nz]

        This lets one FFT batch use a different Toeplitz kernel for every
        contrast without duplicating the kernel across coils.
        """
        if self.toeplitz_kernels is None or kernel_indices is None:
            return None

        active_indices = set(
            int(kernel_indices[b]) for b in range(kernel_indices.shape[0])
        )

        for ki in active_indices:
            if ki in self._kernel_gpu_cache:
                continue

            first = self.toeplitz_kernels[ki][0]
            kernel_shape = tuple(first.shape)
            kernel_dtype = first.dtype

            stacked = cp.empty(
                (self.ncontrast, *kernel_shape),
                dtype=kernel_dtype,
            )

            for c in range(self.ncontrast):
                kernel_c = self.toeplitz_kernels[ki][c]
                if tuple(kernel_c.shape) != kernel_shape:
                    raise ValueError(
                        "All contrast kernels must have the same shape."
                    )

                # Training stores kernels on CPU as NumPy arrays.  .set()
                # copies directly into the already allocated GPU slice and
                # avoids creating five temporary GPU kernel arrays.
                if isinstance(kernel_c, cp.ndarray):
                    cp.copyto(stacked[c], kernel_c)
                else:
                    stacked[c].set(kernel_c)

            self._kernel_gpu_cache[ki] = stacked

        return {
            ki: self._kernel_gpu_cache[ki]
            for ki in active_indices
        }


    def _get_fast_dc_workspace(self, spatial_shape):
        """Create the shared SENSE/FFT workspace once, then reuse it."""
        if self.toeplitz_operators is None:
            return None

        if self._fast_dc_workspace is None:
            # 0 means: use the same coil batching as the existing operator.
            coil_batch = int(getattr(self.args, "cg_coil_batch", 0))
            if coil_batch <= 0:
                coil_batch = int(
                    getattr(self.toeplitz_operators[0], "n_trans", 1)
                )

            self._fast_dc_workspace = (
                ssdu_dc.BatchedContrastSenseToeplitzWorkspace(
                    smaps=self.toeplitz_operators[0].smaps,
                    ncontrast=self.ncontrast,
                    spatial_shape=spatial_shape,
                    coil_batch=coil_batch,
                )
            )

        return self._fast_dc_workspace


    def forward(
        self,
        input_x,
        trn_mask,
        loss_mask,
        kernel_indices=None,
    ):
        """
        input_x    : [B, 2*C, Nx, Ny, Nz]   (3D) or [B, 2*C, H, W]   (2D)
        trn_mask   : [B, C, Nsample]
        loss_mask  : [B, C, Nsample]
        kernel_indices : [B]
        """

        active_kernels = self._get_active_kernels(kernel_indices)
        fast_dc_workspace = self._get_fast_dc_workspace(input_x.shape[2:])

        # Shared backward CG warm starts across unroll blocks.
        # One cache per batch element, reset for each forward pass.
        if self.training and getattr(self.args, 'backward_cg_warmstart', False):
            backward_caches = [
                {'x0': None} for _ in range(input_x.shape[0])
            ]
        else:
            backward_caches = None

        for block_args in self.cg_args:
            block_args.backward_cg_caches = backward_caches

        # ==============================================================
        # Unroll loop
        #
        # Key insight from profiling:
        #
        #   ResNet:  ~0.07s per unroll   — tiny, cheaply recomputed
        #   CG:      ~6.1s  per unroll   — runs under torch.no_grad()
        #                                  via ImplicitCGFunction, so
        #                                  checkpoint CANNOT skip it
        #
        # Original code wrapped [ResNet + CG] together inside checkpoint,
        # causing CG to be re-run 7 more times during backward (~44s waste).
        #
        # Fix: checkpoint ONLY the ResNet (the only part that has an
        # autograd graph worth saving/recomputing). Run CG outside
        # the checkpoint boundary so it executes exactly once per unroll.
        #
        # Result: backward recomputes only 7 × ResNet (~0.5s) instead of
        # 7 × [ResNet + 5 × CG] (~44s).  Expected saving: ~43s/epoch.
        # ==============================================================

        def run_resnet(x):
            """Checkpoint boundary: only the ResNet."""
            return self.regularizer(x)

        for block_args in self.cg_args:
            block_args.cg_record = self.training
            
        x = input_x

        for i in range(self.args.nb_unroll_blocks):

            # ----------------------------------------------------------
            # Step 1: ResNet regularization (checkpointed in 3D training)
            # ----------------------------------------------------------

            if self.ndim == 3 and self.training:
                regularized = checkpoint(
                    run_resnet, x, use_reentrant=False
                )
            else:
                regularized = self.regularizer(x)

            # ----------------------------------------------------------
            # Step 2: form RHS
            # ----------------------------------------------------------

            rhs = input_x + self.lam * regularized

            # ----------------------------------------------------------
            # Step 3: CG data consistency with contrasts in parallel
            #
            # The linear systems remain independent contrast-by-contrast.
            # Only the expensive SENSE Toeplitz FFT applications are batched.
            # ----------------------------------------------------------

            x = ssdu_dc.dc_block_batched_contrasts(
                rhs=rhs,
                operators=self.operators,
                mask=trn_mask,
                mu=self.lam,
                args=self.cg_args[i],
                kernels_cp_dict=active_kernels,
                kernel_indices=kernel_indices,
                x0=x,
                ncontrast=self.ncontrast,
                workspace=fast_dc_workspace,
            )

        # ==============================================================
        # Final ResNet pass — no CG after this.
        #
        # The unroll loop ends with CG, which enforces data consistency
        # but reintroduces spiral undersampling artifacts (rings).
        # A final ResNet application suppresses those artifacts without
        # a subsequent CG step pulling them back in.
        #
        # This changes the unroll structure from:
        #   (ResNet → CG) × N   → last output has CG artifacts
        # to:
        #   (ResNet → CG) × N → ResNet   → last output is regularized
        # ==============================================================

        if self.ndim == 3 and self.training:
            x = checkpoint(run_resnet, x, use_reentrant=False)
        else:
            x = self.regularizer(x)

        # ==============================================================
        # SSDU k-space loss
        #
        # Training: return a closure that computes the loss one contrast
        # at a time — avoids the 3.59 GiB torch.stack allocation.
        # Eval:     stack as before (no grad graph, memory pressure low).
        # ==============================================================

        if self.training:

            def kspace_loss_fn(ref_kspace, loss_fn):
                total_loss = None
                for c in range(self.ncontrast):
                    x_c      = x[:, 2 * c : 2 * c + 2, ...]
                    kspace_c = ssdu_dc.SSDU_kspace_transform(
                        x_c, self.operators[c], loss_mask[:, c, :]
                    )
                    ref_c    = ref_kspace[:, c, ...]
                    loss_c   = loss_fn(kspace_c, ref_c)
                    total_loss = (
                        loss_c if total_loss is None
                        else total_loss + loss_c
                    )
                    del kspace_c
                return total_loss / self.ncontrast

            return x, self.lam, kspace_loss_fn

        else:

            kspace_outputs = []
            for c in range(self.ncontrast):
                x_c = x[:, 2 * c : 2 * c + 2, ...]
                kspace_c = ssdu_dc.SSDU_kspace_transform(
                    x_c, self.operators[c], loss_mask[:, c, :]
                )
                kspace_outputs.append(kspace_c)
            return x, self.lam, torch.stack(kspace_outputs, dim=1)