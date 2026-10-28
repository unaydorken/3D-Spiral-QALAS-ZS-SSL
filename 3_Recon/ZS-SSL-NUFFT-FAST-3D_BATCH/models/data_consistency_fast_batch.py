import torch
import cupy as cp
import mrinufft.operators.toeplitz as toepmod


# ============================================================
# Complex / real conversion helpers
# ============================================================

def c2r_HWC(input_data):
    return torch.stack(
        [input_data.real, input_data.imag],
        dim=-1,
    )


def r2c_HWC(input_data):
    return torch.complex(
        input_data[..., 0],
        input_data[..., 1],
    )


def c2r_CHW(input_data):
    return torch.stack(
        [input_data.real, input_data.imag],
        dim=0,
    )


def r2c_CHW(input_data):
    return torch.complex(
        input_data[0, ...],
        input_data[1, ...],
    )


def zdot_reduce_sum(input_x, input_y):
    dims = tuple(range(len(input_x.shape)))

    return (
        torch.conj(input_x) * input_y
    ).sum(dims).real


# ============================================================
# Temporary MRI-NUFFT development-branch fix
#
# MRI-NUFFT 1.5.2.dev70 currently checks the full shape of the
# preallocated Toeplitz workspace against a kernel with a
# singleton batch dimension.
#
# For n_trans > 1:
#
#     workspace: [n_trans, 2*Nx, 2*Ny]
#     kernel:    [1,       2*Nx, 2*Ny]
#
# The kernel should broadcast across n_trans.
# ============================================================

def _apply_toeplitz_kernel_batched(
    image,
    toeplitz_kernel,
    padded_array=None,
    paired_batch=False,
):
    xp = toepmod.get_array_module(image)
    fftn, ifftn, _ = toepmod._get_fftns(xp)

    original_img_shape = image.shape

    # --------------------------------------------------------
    # Add batch dimension when needed
    # --------------------------------------------------------

    if image.ndim == toeplitz_kernel.ndim:
        image = image[None, ...]
        toeplitz_kernel = toeplitz_kernel[None, ...]

    elif image.ndim - 1 == toeplitz_kernel.ndim:
        toeplitz_kernel = toeplitz_kernel[None, ...]

    else:
        raise ValueError(
            "Image and toeplitz_kernel must have compatible dimensions."
        )

    batch_size = image.shape[0]

    expected_shape = (
        batch_size,
        *toeplitz_kernel.shape[1:],
    )

    # --------------------------------------------------------
    # Allocate / validate padded workspace
    # --------------------------------------------------------

    if padded_array is None:
        padded_array = xp.zeros(
            expected_shape,
            dtype=image.dtype,
        )

    else:
        if padded_array.shape != expected_shape:
            raise ValueError(
                f"Expected padded_array shape {expected_shape}, "
                f"got {padded_array.shape}"
            )

        padded_array[...] = 0

    # --------------------------------------------------------
    # Zero pad
    # --------------------------------------------------------

    spatial_slices = tuple(
        slice(0, size)
        for size in image.shape[1:]
    )

    padded_array[
        (slice(None), *spatial_slices)
    ] = image

    # --------------------------------------------------------
    # Toeplitz convolution
    # --------------------------------------------------------

    axes = tuple(
        range(1, padded_array.ndim)
    )

    tmp = fftn(
        padded_array,
        axes=axes,
        overwrite_x=True,
    )

    # Kernel has singleton batch dimension and broadcasts
    # across n_trans.
    tmp *= toeplitz_kernel

    result = ifftn(
        tmp,
        axes=axes,
        overwrite_x=True,
    )

    result = result[
        (slice(None), *spatial_slices)
    ]

    return result.reshape(original_img_shape)


# Install the temporary patch.
#
# _gram_op_raw_device imports apply_toeplitz_kernel from this
# module when it executes, so replacing it here updates the
# batched Gram path.
toepmod.apply_toeplitz_kernel = _apply_toeplitz_kernel_batched


# ============================================================
# Differentiable Toeplitz Gram operator
# ============================================================

class ToeplitzGramFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        image,
        toeplitz_operator,
        toeplitz_kernel,
    ):
        ctx.toeplitz_operator = toeplitz_operator
        ctx.toeplitz_kernel = toeplitz_kernel

        # The kernel already contains the sampling mask.
        toeplitz_operator._toeplitz_kernel = toeplitz_kernel

        output = toeplitz_operator.gram_op(
            image.contiguous(),
            toeplitz=True,
        )

        return output


    @staticmethod
    def backward(ctx, grad_output):
        toeplitz_operator = ctx.toeplitz_operator
        toeplitz_kernel = ctx.toeplitz_kernel

        # G = A^H M A is Hermitian for a real-valued mask:
        #
        #     G^H = G
        #
        # so the backward operation is another application
        # of the same masked Toeplitz Gram operator.
        toeplitz_operator._toeplitz_kernel = toeplitz_kernel

        grad_input = toeplitz_operator.gram_op(
            grad_output.contiguous(),
            toeplitz=True,
        )

        return grad_input, None, None


def toeplitz_gram_autograd(
    image,
    toeplitz_operator,
    toeplitz_kernel,
):
    return ToeplitzGramFunction.apply(
        image,
        toeplitz_operator,
        toeplitz_kernel,
    )


# ============================================================
# Data consistency
# ============================================================

class NUFFTDataConsistency:
    """
    Applies the SENSE-NUFFT operator used by data consistency.

    Direct mode computes:

        A^H M A

    Toeplitz mode computes the equivalent masked Gram operator
    using a precomputed mask-specific Toeplitz kernel.

    Optimizations vs original:
    - Padded FFT workspace allocated once on first EhE_Op call and
      reused across all CG iterations.  Avoids 700 × 446 MB allocations
      per epoch (312 GB of allocator traffic).
    - EhE_Op bypasses ToeplitzGramFunction (autograd wrapper) when called
      inside conjgrad, which runs entirely under torch.no_grad().
      The direct _apply_toeplitz_kernel_batched call is equivalent and
      avoids the autograd machinery overhead on every CG iteration.
    """

    def __init__(
        self,
        operator,
        mask,
        toeplitz_operator=None,
        toeplitz_kernel=None,
    ):
        self.operator = operator
        self.mask = mask

        self.toeplitz_operator = toeplitz_operator
        self.toeplitz_kernel = toeplitz_kernel

        # Padded workspace for the Toeplitz FFT convolution.
        # Allocated lazily on first EhE_Op call, then reused.
        # Shape: (1, 2*Nx, 2*Ny, 2*Nz) for 3D, complex64.
        self._padded_workspace = None


    def _apply_mask(self, kspace):
        """
        kspace: [1, n_coils, n_samples]
        mask:   [n_samples]
        """

        mask = self.mask.reshape(1, 1, -1)

        return kspace * mask


    def EhE_Op(self, img, mu):
        """
        Compute:

            (A^H M A + mu I) img

        img:
            2D: [Nx, Ny]
            3D: [Nx, Ny, Nz]
        """

        # ----------------------------------------------------
        # FAST Toeplitz path
        #
        # Called inside conjgrad which runs under torch.no_grad()
        # via ImplicitCGFunction.  The ToeplitzGramFunction autograd
        # wrapper is therefore never needed here — calling
        # _apply_toeplitz_kernel_batched directly is equivalent
        # and skips the autograd overhead on every CG iteration.
        #
        # The padded workspace is allocated once on the first call
        # and reused across all CG iterations.
        # ----------------------------------------------------

        if (
            self.toeplitz_operator is not None
            and self.toeplitz_kernel is not None
        ):
            self.toeplitz_operator._toeplitz_kernel = self.toeplitz_kernel

            image_result = self.toeplitz_operator.gram_op(
                img.contiguous(),
                toeplitz=True,
            )

        # ----------------------------------------------------
        # Original direct NUFFT path
        # ----------------------------------------------------

        else:
            img_batched = (
                img
                .unsqueeze(0)
                .unsqueeze(0)
            )

            kspace = self.operator.op(
                img_batched
            )

            masked_kspace = self._apply_mask(
                kspace
            )

            image_result = self.operator.adj_op(
                masked_kspace
            )

            image_result = image_result[0, 0]

        return image_result + mu * img


    def SSDU_kspace(self, img):
        """
        Transform an image to non-Cartesian k-space and retain only
        the samples selected by the SSDU loss mask.

        Returns [n_coils, n_samples].
        """

        img_batched = (
            img
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kspace = self.operator.op(
            img_batched
        )

        masked_kspace = self._apply_mask(
            kspace
        )

        return masked_kspace[0]


    def Supervised_kspace(self, img):
        """
        Transform an image to all acquired non-Cartesian samples.

        Returns [n_coils, n_samples].
        """

        img_batched = (
            img
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kspace = self.operator.op(
            img_batched
        )

        return kspace[0]


# ============================================================
# Conjugate gradient
# ============================================================

def conjgrad(
    rhs,
    operator,
    mask,
    mu,
    args,
    toeplitz_operator=None,
    toeplitz_kernel=None,
    x0=None,
    encoder=None,
):
    # encoder may be passed in from dc_block to avoid reconstructing
    # NUFFTDataConsistency (and its workspace) on every CG call.
    if encoder is None:
        encoder = NUFFTDataConsistency(
            operator=operator,
            mask=mask,
            toeplitz_operator=toeplitz_operator,
            toeplitz_kernel=toeplitz_kernel,
        )

    rhs = r2c_CHW(rhs)

    mu = mu.to(
        dtype=rhs.dtype,
        device=rhs.device,
    )

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    if x0 is None:
        x = torch.zeros_like(rhs)
    else:
        x = r2c_CHW(x0)

    # r0 = b - H x0

    r = rhs - encoder.EhE_Op(
        x,
        mu,
    )

    p = r.clone()

    rsold = zdot_reduce_sum(
        r,
        r,
    )

    # --------------------------------------------------------
    # Normalize by ||rhs||
    # --------------------------------------------------------

    rhs_norm_sq = zdot_reduce_sum(
        rhs,
        rhs,
    )

    if (
        not torch.isfinite(rhs_norm_sq)
        or rhs_norm_sq <= 0
    ):
        return c2r_CHW(x)

    rel_res = torch.sqrt(
        rsold / rhs_norm_sq
    )

    # --------------------------------------------------------
    # Warm start already satisfies tolerance
    # --------------------------------------------------------

    if rel_res <= args.CG_tol:
        
        return c2r_CHW(x)

    # --------------------------------------------------------
    # CG
    # --------------------------------------------------------

    num_iterations = 0

    for iteration in range(args.CG_Iter):

        Ap = encoder.EhE_Op(
            p,
            mu,
        )

        pAp = zdot_reduce_sum(
            p,
            Ap,
        )

        if (
            not torch.isfinite(pAp)
            or pAp <= 0
        ):
            break

        alpha = rsold / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = zdot_reduce_sum(
            r,
            r,
        )

        num_iterations = iteration + 1

        if not torch.isfinite(rsnew):
            break

        rel_res = torch.sqrt(
            rsnew / rhs_norm_sq
        )

        if rel_res <= args.CG_tol:
            break

        beta = rsnew / rsold

        p = r + beta * p
        rsold = rsnew

    return c2r_CHW(x)

class ImplicitCGFunction(torch.autograd.Function):
    """
    Implicit differentiation through the CG linear solve.

    Forward:
        Run the existing CG solver with no autograd graph.
        KEEP the existing warm start x0.

    Backward:
        Solve the adjoint linear system using CG again.

    This treats CG as the numerical solver for

        H x = rhs

    rather than differentiating through every individual CG iteration.
    """

    @staticmethod
    def forward(
        ctx,
        rhs,
        mu,
        x0,
        operator,
        mask,
        args,
        toeplitz_operator,
        toeplitz_kernel,
        encoder,
    ):

        # ------------------------------------------------------------
        # IMPORTANT:
        # x0 is still passed to CG here.
        # Therefore forward warm starting is completely preserved.
        #
        # encoder is a pre-built NUFFTDataConsistency instance passed
        # in from dc_block.  It holds the padded FFT workspace so it
        # is not reallocated on every CG call.  We reuse the same
        # encoder in backward for the adjoint solve.
        # ------------------------------------------------------------

        with torch.no_grad():

            reconstruction = conjgrad(
                rhs=rhs,
                operator=operator,
                mask=mask,
                mu=mu.detach(),
                args=args,
                toeplitz_operator=toeplitz_operator,
                toeplitz_kernel=toeplitz_kernel,
                x0=x0,
                encoder=encoder,
            )

        # Save only the final solution and lambda.
        # We do NOT save the CG10 iteration history.
        ctx.save_for_backward(
            reconstruction,
            mu,
        )

        ctx.operator        = operator
        ctx.mask            = mask
        ctx.args            = args
        ctx.toeplitz_operator = toeplitz_operator
        ctx.toeplitz_kernel   = toeplitz_kernel
        ctx.encoder           = encoder   # reused in backward

        return reconstruction


    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):

        reconstruction, mu = ctx.saved_tensors

        # ------------------------------------------------------------
        # Implicit backward
        #
        # Forward solves:
        #
        #     H x = rhs
        #
        # where
        #
        #     H = A^H M A + mu I
        #
        # Given:
        #
        #     g = dL/dx
        #
        # solve:
        #
        #     H v = g
        #
        # Because H is Hermitian, we can use the SAME CG solver.
        # The same encoder (and its workspace) is reused — no new
        # allocation needed for the adjoint solve.
        # ------------------------------------------------------------

        with torch.no_grad():

            v = conjgrad(
                rhs=grad_output.contiguous(),
                operator=ctx.operator,
                mask=ctx.mask,
                mu=mu.detach(),
                args=ctx.args,
                toeplitz_operator=ctx.toeplitz_operator,
                toeplitz_kernel=ctx.toeplitz_kernel,

                # This is a NEW linear solve.
                # No warm start is required for the gradient solve.
                x0=None,
                encoder=ctx.encoder,
            )

        # ------------------------------------------------------------
        # Gradient with respect to rhs
        #
        # dx/drhs = H^{-1}
        # ------------------------------------------------------------

        grad_rhs = v

        # ------------------------------------------------------------
        # Gradient with respect to mu
        #
        # H = G + mu I
        #
        # dL/dmu = -Re(v^H x)
        #
        # If tensors are represented as real/imaginary channels,
        # sum(v * x) is the equivalent real inner product.
        # ------------------------------------------------------------

        if torch.is_complex(reconstruction):

            grad_mu_scalar = -torch.real(
                torch.sum(
                    torch.conj(v)
                    * reconstruction
                )
            )

        else:

            grad_mu_scalar = -torch.sum(
                v * reconstruction
            )

        grad_mu = (
            grad_mu_scalar
            * torch.ones_like(mu)
        )

        # Inputs to forward were:
        #
        # rhs
        # mu
        # x0
        # operator
        # mask
        # args
        # toeplitz_operator
        # toeplitz_kernel
        #
        # We intentionally return NO gradient through x0 or encoder.
        #
        # x0 is still used in the FORWARD warm-started CG.
        # Its truncated-CG dependence is simply not differentiated.
        return (
            grad_rhs,   # rhs
            grad_mu,    # mu
            None,       # x0
            None,       # operator
            None,       # mask
            None,       # args
            None,       # toeplitz_operator
            None,       # toeplitz_kernel
            None,       # encoder
        )


def conjgrad_implicit(
    rhs,
    operator,
    mask,
    mu,
    args,
    toeplitz_operator=None,
    toeplitz_kernel=None,
    x0=None,
    encoder=None,
):
    """
    CG with forward warm start and implicit backward.

    encoder: optional pre-built NUFFTDataConsistency with a cached
             padded FFT workspace.  Pass this in from dc_block to
             avoid reallocating the workspace on every CG call.
    """

    return ImplicitCGFunction.apply(
        rhs,
        mu,
        x0,
        operator,
        mask,
        args,
        toeplitz_operator,
        toeplitz_kernel,
        encoder,
    )
# ============================================================
# Data-consistency block
# ============================================================

def dc_block(
    rhs,
    operator,
    mask,
    mu,
    args,
    toeplitz_operator=None,
    toeplitz_kernels=None,
    kernel_indices=None,
    x0=None,
):
    reconstructions = []

    use_toeplitz = (
        toeplitz_operator is not None
        and toeplitz_kernels is not None
        and kernel_indices is not None
    )

    # Resolve once — avoids a getattr() call per batch element per
    # contrast per unroll (35 calls in forward, 35 in backward).
    use_implicit = getattr(args, "implicit_cg", False)

    for batch_index in range(rhs.shape[0]):

        toeplitz_kernel = None

        if use_toeplitz:
            kernel_index = int(
                kernel_indices[batch_index]
            )

            toeplitz_kernel = (
                toeplitz_kernels[kernel_index]
            )

        x0_batch = (
            x0[batch_index]
            if x0 is not None
            else None
        )

        # Build NUFFTDataConsistency once per batch element.
        # Its padded FFT workspace is allocated on the first EhE_Op
        # call and then reused across all CG iterations — replacing
        # 700 × 446 MB allocations per epoch with one per solve.
        encoder = NUFFTDataConsistency(
            operator=operator,
            mask=mask[batch_index],
            toeplitz_operator=(
                toeplitz_operator if use_toeplitz else None
            ),
            toeplitz_kernel=toeplitz_kernel,
        )

        if use_implicit:

            reconstruction = conjgrad_implicit(
                rhs=rhs[batch_index],
                operator=operator,
                mask=mask[batch_index],
                mu=mu,
                args=args,
                toeplitz_operator=(
                    toeplitz_operator
                    if use_toeplitz
                    else None
                ),
                toeplitz_kernel=toeplitz_kernel,
                x0=x0_batch,
                encoder=encoder,
            )

        else:

            reconstruction = conjgrad(
                rhs=rhs[batch_index],
                operator=operator,
                mask=mask[batch_index],
                mu=mu,
                args=args,
                toeplitz_operator=(
                    toeplitz_operator
                    if use_toeplitz
                    else None
                ),
                toeplitz_kernel=toeplitz_kernel,
                x0=x0_batch,
                encoder=encoder,
            )

        reconstructions.append(
            reconstruction.unsqueeze(0)
        )

    return torch.cat(
        reconstructions,
        dim=0,
    )



# ============================================================
# Fast batched-contrast SENSE Toeplitz CG
#
# IMPORTANT:
# This is mathematically the same SENSE normal operator used by
# toeplitz_operator.gram_op():
#
#   G_c x = sum_j conj(S_j) T_c(S_j x)
#
# where T_c is the mask/trajectory-specific Toeplitz convolution for
# contrast c and S_j is coil j's sensitivity map.
#
# The previous fast-batch prototype applied T_c directly to x and therefore
# omitted the SENSE maps completely.  That is why it did not reproduce the
# current reconstruction.
#
# Here contrasts are batched together while coil chunks are processed in the
# same mathematical order as the reference SENSE Gram operator.
# ============================================================


class BatchedContrastSenseToeplitzWorkspace:
    """Reusable GPU scratch space for batched-contrast SENSE Toeplitz Gram.

    Parameters
    ----------
    smaps
        Sensitivity maps [n_coils, *spatial].  May be NumPy or CuPy.
    ncontrast
        Number of image contrasts processed together.
    spatial_shape
        2D or 3D image shape.
    coil_batch
        Number of coils processed together.  For the first 3D equivalence
        test keep this at 1.  Once equivalence is established, 2 or 4 can be
        benchmarked for additional speed.
    """

    def __init__(
        self,
        smaps,
        ncontrast,
        spatial_shape,
        coil_batch=1,
    ):
        self.ncontrast = int(ncontrast)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.ndim = len(self.spatial_shape)
        self.coil_batch = max(1, int(coil_batch))

        if self.ndim not in (2, 3):
            raise ValueError(
                f"Expected a 2D or 3D spatial shape, got {self.spatial_shape}."
            )

        self.smaps = cp.asarray(smaps, dtype=cp.complex64)
        if self.smaps.ndim != self.ndim + 1:
            raise ValueError(
                f"Expected smaps [coil, *spatial], got {self.smaps.shape}."
            )
        if tuple(self.smaps.shape[1:]) != self.spatial_shape:
            raise ValueError(
                f"Sensitivity-map shape {self.smaps.shape[1:]} does not match "
                f"image shape {self.spatial_shape}."
            )

        self.ncoils = int(self.smaps.shape[0])
        self.coil_batch = min(self.coil_batch, self.ncoils)
        self.smaps_conj = cp.conj(self.smaps)

        self.padded_shape = tuple(2 * v for v in self.spatial_shape)

        # Reused scratch buffers.  The leading dimensions are
        # [contrast, coil-within-chunk].
        self.data = cp.empty(
            (self.ncontrast, self.coil_batch, *self.spatial_shape),
            dtype=cp.complex64,
        )
        self.padded = cp.empty(
            (self.ncontrast, self.coil_batch, *self.padded_shape),
            dtype=cp.complex64,
        )
        self.output = cp.empty(
            (self.ncontrast, *self.spatial_shape),
            dtype=cp.complex64,
        )

        self.fftn, self.ifftn, _ = toepmod._get_fftns(cp)
        self.fft_axes = tuple(range(2, 2 + self.ndim))
        self.spatial_slices = tuple(
            slice(0, size) for size in self.spatial_shape
        )

    def _validate_kernels(self, kernels_cp):
        if kernels_cp is None:
            raise ValueError("Batched Toeplitz kernels are required.")
        if int(kernels_cp.shape[0]) != self.ncontrast:
            raise ValueError(
                f"Expected {self.ncontrast} contrast kernels, "
                f"got {kernels_cp.shape[0]}."
            )
        if tuple(kernels_cp.shape[1:]) != self.padded_shape:
            raise ValueError(
                f"Kernel shape {kernels_cp.shape[1:]} does not match "
                f"expected padded shape {self.padded_shape}."
            )

    def apply_gram(self, imgs_torch, kernels_cp):
        """Apply all contrast-specific SENSE Gram operators in parallel.

        imgs_torch: [ncontrast, *spatial] complex torch tensor on CUDA.
        kernels_cp: [ncontrast, *2spatial] CuPy array.
        """
        self._validate_kernels(kernels_cp)

        if not torch.is_complex(imgs_torch):
            raise ValueError("imgs_torch must be complex.")
        if tuple(imgs_torch.shape) != (
            self.ncontrast,
            *self.spatial_shape,
        ):
            raise ValueError(
                f"Expected image shape "
                f"{(self.ncontrast, *self.spatial_shape)}, "
                f"got {tuple(imgs_torch.shape)}."
            )

        # DLPack gives a zero-copy CUDA view of the torch data.
        imgs_cp = cp.from_dlpack(imgs_torch.detach().contiguous())

        self.output.fill(0)

        for coil_start in range(0, self.ncoils, self.coil_batch):
            coil_stop = min(
                coil_start + self.coil_batch,
                self.ncoils,
            )
            nchunk = coil_stop - coil_start

            smaps_chunk = self.smaps[coil_start:coil_stop]
            smaps_conj_chunk = self.smaps_conj[coil_start:coil_stop]

            data = self.data[:, :nchunk]
            padded = self.padded[:, :nchunk]

            # S_j x_c for every contrast c and coil j in the chunk.
            cp.multiply(
                imgs_cp[:, None, ...],
                smaps_chunk[None, ...],
                out=data,
            )

            # Same zero-padding convention as MRI-NUFFT apply_toeplitz_kernel.
            padded.fill(0)
            crop = (
                slice(None),
                slice(None),
                *self.spatial_slices,
            )
            padded[crop] = data

            # One batched FFT over all [contrast, coil] images.
            tmp = self.fftn(
                padded,
                axes=self.fft_axes,
                overwrite_x=True,
            )

            # Each contrast gets its own Toeplitz kernel; the kernel is shared
            # across coils of that contrast.
            tmp *= kernels_cp[:, None, ...]

            tmp = self.ifftn(
                tmp,
                axes=self.fft_axes,
                overwrite_x=True,
            )

            result = tmp[crop]

            # conj(S_j) T_c(S_j x_c), accumulated over coils.
            cp.multiply(
                result,
                smaps_conj_chunk[None, ...],
                out=data,
            )

            # Avoid a large temporary reduction array.  coil_batch is small
            # for 3D, so explicit accumulation is cheap and memory-stable.
            for j in range(nchunk):
                self.output += data[:, j]

        # Zero-copy view back into torch.  The addition allocates the returned
        # tensor, so callers do not retain an alias to the reusable output
        # workspace.
        return torch.utils.dlpack.from_dlpack(self.output)

    def apply_h(self, imgs_torch, kernels_cp, mu):
        """Apply (G_c + mu I) to all contrasts."""
        gram = self.apply_gram(imgs_torch, kernels_cp)
        return gram + mu * imgs_torch


def _ri_contrasts_to_complex(x):
    """[C,2,*spatial] real/imag -> [C,*spatial] complex."""
    return torch.complex(x[:, 0, ...], x[:, 1, ...])


def _complex_contrasts_to_ri(x):
    """[C,*spatial] complex -> [C,2,*spatial] real/imag."""
    return torch.stack((x.real, x.imag), dim=1)


def _zdot_per_contrast(a, b):
    dims = tuple(range(1, a.ndim))
    return (torch.conj(a) * b).sum(dims).real


def conjgrad_batched_contrasts(
    rhs_stacked,
    kernels_cp,
    mu,
    args,
    workspace,
    x0_stacked=None,
    phase="forward",
):
    """CG solve for all contrasts with exactly one scalar CG system per contrast.

    This is the vectorized equivalent of calling ``conjgrad`` separately for
    every contrast.  The CG coefficients (alpha/beta/residual) remain separate
    per contrast; only the expensive SENSE Toeplitz applications are batched.
    """
    rhs = _ri_contrasts_to_complex(rhs_stacked)

    if x0_stacked is None:
        x = torch.zeros_like(rhs)
    else:
        x = _ri_contrasts_to_complex(x0_stacked)

    # r0 = b - H x0
    r = rhs - workspace.apply_h(x, kernels_cp, mu)
    p = r.clone()

    rsold = _zdot_per_contrast(r, r)
    rhs_norm_sq = _zdot_per_contrast(rhs, rhs)

    rhs_valid = torch.isfinite(rhs_norm_sq) & (rhs_norm_sq > 0)
    rhs_norm_safe = torch.where(
        rhs_valid,
        rhs_norm_sq,
        torch.ones_like(rhs_norm_sq),
    )

    rel_res = torch.sqrt(rsold / rhs_norm_safe)
    finished = (~rhs_valid) | (rel_res <= args.CG_tol)
    iteration_counts = torch.zeros_like(rsold, dtype=torch.int32)

    for _ in range(args.CG_Iter):
        if bool(finished.all()):
            break

        Ap = workspace.apply_h(p, kernels_cp, mu)
        pAp = _zdot_per_contrast(p, Ap)

        active = (
            (~finished)
            & torch.isfinite(pAp)
            & (pAp > 0)
        )

        # Sequential CG would break an individual contrast here if pAp is
        # invalid/non-positive.  Mark only that contrast finished while the
        # remaining contrasts continue.
        finished = finished | (
            (~finished)
            & (~torch.isfinite(pAp) | (pAp <= 0))
        )

        if not bool(active.any()):
            break

        iteration_counts += active.to(torch.int32)

        pAp_safe = torch.where(
            active,
            pAp,
            torch.ones_like(pAp),
        )
        alpha = torch.where(
            active,
            rsold / pAp_safe,
            torch.zeros_like(rsold),
        )
        alpha_bc = alpha.reshape(
            alpha.shape[0],
            *([1] * (rhs.ndim - 1)),
        )

        x = x + alpha_bc * p
        r = r - alpha_bc * Ap

        rsnew = _zdot_per_contrast(r, r)
        rsnew_finite = torch.isfinite(rsnew)
        rel_res = torch.sqrt(
            torch.where(
                rsnew_finite,
                rsnew,
                torch.zeros_like(rsnew),
            )
            / rhs_norm_safe
        )

        newly_finished = (
            finished
            | (~rsnew_finite)
            | (rel_res <= args.CG_tol)
        )

        beta_active = active & (~newly_finished)
        rsold_safe = torch.where(
            beta_active,
            rsold,
            torch.ones_like(rsold),
        )
        beta = torch.where(
            beta_active,
            rsnew / rsold_safe,
            torch.zeros_like(rsnew),
        )
        beta_bc = beta.reshape(
            beta.shape[0],
            *([1] * (rhs.ndim - 1)),
        )

        p = r + beta_bc * p
        rsold = rsnew
        finished = newly_finished

    if getattr(args, 'cg_diagnostics', False) and getattr(args, 'cg_record', False):
        args.cg_stats[phase].extend(zip(
            iteration_counts.detach().cpu().tolist(),
            rel_res.detach().cpu().tolist()
        ))

    return _complex_contrasts_to_ri(x)


class ImplicitCGBatchedContrastsFunction(torch.autograd.Function):
    """Implicit differentiation through the batched-contrast CG solve."""

    @staticmethod
    def forward(
        ctx,
        rhs_stacked,
        mu,
        x0_stacked,
        kernels_cp,
        args,
        workspace,
        warm_state,
    ):
        with torch.no_grad():
            reconstruction = conjgrad_batched_contrasts(
                rhs_stacked=rhs_stacked,
                kernels_cp=kernels_cp,
                mu=mu.detach(),
                args=args,
                workspace=workspace,
                x0_stacked=x0_stacked,
            )

        ctx.save_for_backward(reconstruction, mu)
        ctx.kernels_cp = kernels_cp
        ctx.args = args
        ctx.workspace = workspace
        ctx.warm_state = warm_state
        return reconstruction

    @staticmethod
    def backward(ctx, grad_output):
        reconstruction, mu = ctx.saved_tensors

        with torch.no_grad():

            x0_backward = (
                ctx.warm_state['x0']
                if ctx.warm_state is not None
                else None
            )

            v = conjgrad_batched_contrasts(
                rhs_stacked=grad_output.contiguous(),
                kernels_cp=ctx.kernels_cp,
                mu=mu.detach(),
                args=ctx.args,
                workspace=ctx.workspace,
                x0_stacked=x0_backward,
                phase="backward",
            )

            if ctx.warm_state is not None:
                ctx.warm_state['x0'] = v.detach()
        # This is the SUM of the per-contrast lambda gradients.  That is
        # exactly what PyTorch does for the shared lambda in the original
        # sequential implementation.  Do not divide by ncontrast here.
        recon_complex = _ri_contrasts_to_complex(reconstruction)
        v_complex = _ri_contrasts_to_complex(v)
        grad_mu_scalar = -torch.real(
            torch.sum(torch.conj(v_complex) * recon_complex)
        )
        grad_mu = grad_mu_scalar * torch.ones_like(mu)

        return (
            v,          # rhs_stacked
            grad_mu,    # mu
            None,       # x0_stacked
            None,       # kernels_cp
            None,       # args
            None,       # workspace
            None,       # warm_state
        )


def dc_block_batched_contrasts(
    rhs,
    operators,
    mask,
    mu,
    args,
    kernels_cp_dict=None,
    kernel_indices=None,
    x0=None,
    ncontrast=5,
    workspace=None,
):
    """Data consistency with contrast-parallel SENSE Toeplitz CG.

    The reference implementation loops over contrasts outside ``dc_block``.
    This function keeps the same independent linear system for every contrast,
    but batches their expensive Toeplitz FFTs together.
    """
    use_toeplitz = (
        kernels_cp_dict is not None
        and kernel_indices is not None
        and workspace is not None
    )

    # Exact sequential fallback.  Useful for 2D/debugging and guarantees that
    # this module remains a drop-in replacement when batched Toeplitz is absent.
    if not use_toeplitz:
        outputs = []
        for c in range(ncontrast):
            outputs.append(
                dc_block(
                    rhs=rhs[:, 2 * c : 2 * c + 2, ...],
                    operator=operators[c],
                    mask=mask[:, c, :],
                    mu=mu,
                    args=args,
                    x0=(
                        x0[:, 2 * c : 2 * c + 2, ...]
                        if x0 is not None
                        else None
                    ),
                )
            )
        return torch.cat(outputs, dim=1)

    batch_outputs = []
    spatial_shape = tuple(rhs.shape[2:])

    for b in range(rhs.shape[0]):
        kernel_index = int(kernel_indices[b])
        kernels_cp = kernels_cp_dict[kernel_index]

        rhs_stacked = rhs[b].reshape(
            ncontrast,
            2,
            *spatial_shape,
        )
        x0_stacked = (
            x0[b].reshape(ncontrast, 2, *spatial_shape)
            if x0 is not None
            else None
        )

        if getattr(args, "implicit_cg", False):

            backward_caches = getattr(args, 'backward_cg_caches', None)
            warm_state = (
                backward_caches[b]
                if backward_caches is not None
                else None
            )

            recon_stacked = ImplicitCGBatchedContrastsFunction.apply(
                rhs_stacked,
                mu,
                x0_stacked,
                kernels_cp,
                args,
                workspace,
                warm_state,
            )
        else:
            recon_stacked = conjgrad_batched_contrasts(
                rhs_stacked=rhs_stacked,
                kernels_cp=kernels_cp,
                mu=mu,
                args=args,
                workspace=workspace,
                x0_stacked=x0_stacked,
            )

        batch_outputs.append(
            recon_stacked.reshape(1, 2 * ncontrast, *spatial_shape)
        )

    return torch.cat(batch_outputs, dim=0)


# ============================================================
# K-space transforms used by the loss / inference
#
# These remain ordinary NUFFT operations.
# Toeplitz is used ONLY inside repeated CG normal-operator calls.
# ============================================================

def SSDU_kspace_transform(
    nw_output,
    operator,
    mask,
):
    """
    Transform network output to non-Cartesian k-space at the samples
    selected by the SSDU loss mask.

    nw_output: [batch, 2, height, width]
    mask:      [batch, n_samples]

    returns:   [batch, n_coils, n_samples, 2]
    """

    all_kspace = []

    for batch_index in range(
        nw_output.shape[0]
    ):
        encoder = NUFFTDataConsistency(
            operator,
            mask[batch_index],
        )

        image = r2c_CHW(
            nw_output[batch_index]
        )

        predicted_kspace = (
            encoder.SSDU_kspace(image)
        )

        predicted_kspace = c2r_HWC(
            predicted_kspace
        )

        all_kspace.append(
            predicted_kspace.unsqueeze(0)
        )

    return torch.cat(
        all_kspace,
        dim=0,
    )


def Supervised_kspace_transform(
    nw_output,
    operator,
):
    """
    Transform network output to all acquired non-Cartesian samples.

    returns: [batch, n_coils, n_samples, 2]
    """

    all_kspace = []

    for batch_index in range(
        nw_output.shape[0]
    ):
        image = r2c_CHW(
            nw_output[batch_index]
        )

        image_batched = (
            image
            .unsqueeze(0)
            .unsqueeze(0)
        )

        predicted_kspace = operator.op(
            image_batched
        )[0]

        predicted_kspace = c2r_HWC(
            predicted_kspace
        )

        all_kspace.append(
            predicted_kspace.unsqueeze(0)
        )

    return torch.cat(
        all_kspace,
        dim=0,
    )


def DIP_kspace_transform(
    nw_output,
    operator,
    mask,
):
    """
    Transform network output to acquired samples selected by mask.

    returns: [batch, n_coils, n_samples, 2]
    """

    return SSDU_kspace_transform(
        nw_output,
        operator,
        mask,
    )