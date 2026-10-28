import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from models.ResNet import ResNet
import models.data_consistency as ssdu_dc
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

        # GPU kernel cache: {kernel_index -> [cp_array_c0, cp_array_c1, ...]}
        # Populated lazily, cleared after each epoch.
        self._kernel_gpu_cache: dict = {}


    def clear_kernel_cache(self):
        """Free cached Toeplitz kernels. Call after optimizer.step()."""
        self._kernel_gpu_cache.clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


    def _get_active_kernels(self, kernel_indices):
        """Upload & cache only the kernels needed by this batch."""
        if self.toeplitz_kernels is None or kernel_indices is None:
            return None

        active_indices = set(
            int(kernel_indices[b]) for b in range(kernel_indices.shape[0])
        )
        for ki in active_indices:
            if ki not in self._kernel_gpu_cache:
                self._kernel_gpu_cache[ki] = [
                    cp.asarray(self.toeplitz_kernels[ki][c])
                    for c in range(self.ncontrast)
                ]
        return {ki: self._kernel_gpu_cache[ki] for ki in active_indices}


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

        x = input_x

        for _ in range(self.args.nb_unroll_blocks):

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
            # Step 3: CG data consistency per contrast
            # ----------------------------------------------------------

            contrast_outputs = []

            for c in range(self.ncontrast):

                rhs_c = rhs[:, 2 * c : 2 * c + 2, ...]
                x0_c  = x[:, 2 * c : 2 * c + 2, ...]

                kernels_c = (
                    {ki: active_kernels[ki][c] for ki in active_kernels}
                    if active_kernels is not None
                    else None
                )

                x_c = ssdu_dc.dc_block(
                    rhs=rhs_c,
                    operator=self.operators[c],
                    mask=trn_mask[:, c, :],
                    mu=self.lam,
                    args=self.args,
                    toeplitz_operator=(
                        self.toeplitz_operators[c]
                        if self.toeplitz_operators is not None
                        else None
                    ),
                    toeplitz_kernels=kernels_c,
                    kernel_indices=kernel_indices,
                    x0=x0_c,
                )

                contrast_outputs.append(x_c)

            x = torch.cat(contrast_outputs, dim=1)

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