import gc
import cupy as cp
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    """
    Training or validation dataset.

    trn_atb:        [repetitions, contrast, *spatial, 2]
    trn_mask:       [repetitions, contrast, n_samples]
    loss_mask:      [repetitions, contrast, n_samples]
    ref_kspace:     [contrast, n_coils, n_samples, 2]
    kernel_indices: [repetitions]
    """

    def __init__(
        self,
        trn_atb,
        trn_mask,
        loss_mask,
        ref_kspace,
        kernel_indices=None,
    ):
        self.trn_atb = trn_atb
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.ref_kspace = ref_kspace

        if kernel_indices is None:
            self.kernel_indices = list(range(len(trn_atb)))
        else:
            if len(kernel_indices) != len(trn_atb):
                raise ValueError(
                    "kernel_indices must have the same length as trn_atb"
                )
            self.kernel_indices = kernel_indices

    def __len__(self):
        return len(self.trn_atb)

    def __getitem__(self, index):

        nw_input = torch.as_tensor(
            self.trn_atb[index],
            dtype=torch.float32,
        )

        nw_trn_mask = torch.as_tensor(
            self.trn_mask[index],
            dtype=torch.float32,
        )

        nw_loss_mask = torch.as_tensor(
            self.loss_mask[index],
            dtype=torch.float32,
        )

        # Full k-space stored only ONCE; apply loss mask here.
        ref_kspace = torch.as_tensor(
            self.ref_kspace,
            dtype=torch.float32,
        )

        ref_kspace = (
            ref_kspace
            * nw_loss_mask[:, None, :, None]
        )

        kernel_index = torch.as_tensor(
            self.kernel_indices[index],
            dtype=torch.long,
        )

        return (
            nw_input,
            nw_trn_mask,
            nw_loss_mask,
            ref_kspace,
            kernel_index,
        )


class Dataset_Inference(torch.utils.data.Dataset):
    """
    Inference dataset.

    trn_atb:   [examples, contrast, *spatial, 2]
    test_mask: [examples, contrast, n_samples]
    loss_mask: [examples, contrast, n_samples]
    """

    def __init__(
        self,
        trn_atb,
        test_mask,
        loss_mask,
    ):
        self.trn_atb = trn_atb
        self.test_mask = test_mask
        self.loss_mask = loss_mask

    def __len__(self):
        return len(self.trn_atb)

    def __getitem__(self, index):

        nw_input = torch.as_tensor(
            self.trn_atb[index],
            dtype=torch.float32,
        )

        nw_test_mask = torch.as_tensor(
            self.test_mask[index],
            dtype=torch.float32,
        )

        nw_loss_mask = torch.as_tensor(
            self.loss_mask[index],
            dtype=torch.float32,
        )

        return (
            nw_input,
            nw_test_mask,
            nw_loss_mask,
        )


class MixL1L2Loss(nn.Module):

    def __init__(
        self,
        eps=1e-6,
        scalar=0.5,
    ):
        super().__init__()
        self.eps = eps
        self.scalar = scalar

    def forward(self, yhat, y):
        difference = yhat - y

        relative_l2 = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(y).clamp_min(self.eps)
        )

        relative_l1 = (
            torch.linalg.vector_norm(difference, ord=1)
            / torch.linalg.vector_norm(y, ord=1).clamp_min(self.eps)
        )

        return self.scalar * relative_l2 + self.scalar * relative_l1


def print_cuda_memory(tag, device):
    if device.type != "cuda":
        return

    torch.cuda.synchronize()
    gb = 1024 ** 3

    allocated      = torch.cuda.memory_allocated()       / gb
    reserved       = torch.cuda.memory_reserved()        / gb
    peak_allocated = torch.cuda.max_memory_allocated()   / gb
    peak_reserved  = torch.cuda.max_memory_reserved()    / gb

    print(
        f"[MEM] {tag}: "
        f"alloc={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB, "
        f"peak_alloc={peak_allocated:.2f} GB, "
        f"peak_reserved={peak_reserved:.2f} GB",
        flush=True,
    )


def train(
    train_loader,
    model,
    loss_fn,
    optimizer,
    device=torch.device("cpu"),
):

    model.train()

    for batch_idx, batch in enumerate(train_loader):

        (
            nw_input,
            nw_trn_mask,
            nw_loss_mask,
            nw_ref_kspace,
            kernel_indices,
        ) = batch

        #print(
        #    f"Using SSDU repetition "
        #    f"{int(kernel_indices[0]) + 1}/{len(train_loader)}",
        #    flush=True,
        #)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        B = nw_input.shape[0]
        C = nw_input.shape[1]
        spatial_shape = nw_input.shape[2:-1]

        permute_order = (
            [0, 1, nw_input.ndim - 1]
            + list(range(2, nw_input.ndim - 1))
        )

        nw_input = (
            nw_input
            .permute(*permute_order)
            .reshape(B, 2 * C, *spatial_shape)
        )

        nw_input       = nw_input.to(device)
        nw_trn_mask    = nw_trn_mask.to(device)
        nw_loss_mask   = nw_loss_mask.to(device)
        nw_ref_kspace  = nw_ref_kspace.to(device)

        optimizer.zero_grad()

        # ------------------------------------------------------------------
        # Forward pass.
        #
        # In training mode UnrolledNet returns a kspace_loss_fn callable
        # instead of the stacked k-space tensor.  This avoids allocating a
        # [B, C, n_coils, n_samples, 2] tensor (the 3.59 GiB OOM crash)
        # by computing and freeing each contrast's k-space one at a time.
        # ------------------------------------------------------------------

        (
            nw_img_output,
            lamdas,
            kspace_loss_fn,
        ) = model(
            nw_input,
            nw_trn_mask,
            nw_loss_mask,
            kernel_indices=kernel_indices,
        )

        # ------------------------------------------------------------------
        # Loss: kspace_loss_fn averages across contrasts without stacking.
        # ref_kspace shape: [B, C, n_coils, n_samples, 2]
        # ------------------------------------------------------------------

        training_loss = kspace_loss_fn(nw_ref_kspace, loss_fn)

        training_loss.backward()

        optimizer.step()

        with torch.no_grad():
            model.lam.clamp_(min=1e-4)

        # ------------------------------------------------------------------
        # Free GPU kernel cache now that gradients have been computed.
        # This prevents accumulation of Toeplitz kernels across epochs.
        # ------------------------------------------------------------------

        model.clear_kernel_cache()

        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        average_loss = training_loss.item()

        #break   # one SSDU repetition per epoch

    if getattr(model.args, 'cg_diagnostics', False):
        model.report_cg_diagnostics()

    return average_loss, lamdas


def validation(
    val_loader,
    model,
    loss_fn,
    device=torch.device("cpu"),
):

    average_loss = 0.0

    model.eval()

    with torch.no_grad():

        for batch in val_loader:

            (
                nw_input,
                nw_trn_mask,
                nw_loss_mask,
                nw_ref_kspace,
                kernel_indices,
            ) = batch

            B = nw_input.shape[0]
            C = nw_input.shape[1]
            spatial_shape = nw_input.shape[2:-1]

            permute_order = (
                [0, 1, nw_input.ndim - 1]
                + list(range(2, nw_input.ndim - 1))
            )

            nw_input = (
                nw_input
                .permute(*permute_order)
                .reshape(B, 2 * C, *spatial_shape)
            )

            nw_input      = nw_input.to(device)
            nw_trn_mask   = nw_trn_mask.to(device)
            nw_loss_mask  = nw_loss_mask.to(device)
            nw_ref_kspace = nw_ref_kspace.to(device)

            # In eval mode the model returns the stacked k-space tensor
            # (no grad graph, so peak memory is much lower).
            _, _, nw_kspace_output = model(
                nw_input,
                nw_trn_mask,
                nw_loss_mask,
                kernel_indices=kernel_indices,
            )

            validation_loss = loss_fn(
                nw_kspace_output,
                nw_ref_kspace,
            )

            average_loss += (
                validation_loss.item()
                / len(val_loader)
            )

        # Free any kernels uploaded during validation.
        model.clear_kernel_cache()

    return average_loss


def test(
    test_loader,
    model,
    device=torch.device("cpu"),
):

    model.eval()
    reconstructions = []

    with torch.no_grad():

        for batch in test_loader:

            (
                nw_input,
                nw_trn_mask,
                nw_loss_mask,
            ) = batch

            B = nw_input.shape[0]
            C = nw_input.shape[1]
            spatial_shape = nw_input.shape[2:-1]

            permute_order = (
                [0, 1, nw_input.ndim - 1]
                + list(range(2, nw_input.ndim - 1))
            )

            nw_input = (
                nw_input
                .permute(*permute_order)
                .reshape(B, 2 * C, *spatial_shape)
            )

            nw_input      = nw_input.to(device)
            nw_trn_mask   = nw_trn_mask.to(device)
            nw_loss_mask  = nw_loss_mask.to(device)

            nw_img_output, _, _ = model(
                nw_input,
                nw_trn_mask,
                nw_loss_mask,
            )

            # [B, 2*C, *spatial] -> [B, C, *spatial, 2]
            nw_img_output = nw_img_output.reshape(
                B, C, 2, *spatial_shape
            )

            output_permute_order = (
                [0, 1]
                + list(range(3, nw_img_output.ndim))
                + [2]
            )

            nw_img_output = nw_img_output.permute(*output_permute_order)

            reconstructions.append(nw_img_output)

    return torch.cat(reconstructions, dim=0)