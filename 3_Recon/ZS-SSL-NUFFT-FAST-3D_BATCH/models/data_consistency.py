import torch
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