import numpy as np
import torch


def split_sample_mask(
    trajectory,
    input_mask,
    rho,
    center_radius_fraction=0.05,
    rng=None,
):
    """
    Split available non-Cartesian samples into training and loss masks.

    Parameters
    ----------
    trajectory : ndarray, [n_samples, 2] or [n_samples, 3]
        Normalized k-space sample coordinates.
    input_mask : ndarray, [n_samples]
        Samples currently available for partitioning.
    rho : float
        Fraction of eligible samples assigned to the loss mask.
    center_radius_fraction : float
        Fraction of the maximum k-space radius protected from removal.
        Central samples always remain in the training mask.
    rng : numpy.random.Generator or None
        Random-number generator for reproducible partitions.

    Returns
    -------
    training_mask : float32 ndarray, [n_samples]
    loss_mask : float32 ndarray, [n_samples]
    """

    trajectory = np.asarray(trajectory)
    input_mask = np.asarray(input_mask).astype(bool)

    if trajectory.ndim != 2 or trajectory.shape[1] not in (2, 3):
        raise ValueError(
            f"Expected trajectory [n_samples, 2] or [n_samples, 3], "
            f"got {trajectory.shape}"
        )

    if input_mask.shape != (trajectory.shape[0],):
        raise ValueError(
            "input_mask and trajectory must have the same sample count: "
            f"{input_mask.shape} versus {trajectory.shape}"
        )

    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be between 0 and 1, got {rho}")

    if rng is None:
        rng = np.random.default_rng()

    radius = np.linalg.norm(trajectory, axis=1)
    center_radius = center_radius_fraction * radius.max()

    # Only available samples outside the protected center may enter the
    # validation/loss set.
    eligible_indices = np.flatnonzero(
        input_mask & (radius > center_radius)
    )

    number_selected = int(round(rho * eligible_indices.size))

    loss_mask = np.zeros(trajectory.shape[0], dtype=np.float32)

    if number_selected > 0:
        selected_indices = rng.choice(
            eligible_indices,
            size=number_selected,
            replace=False,
        )
        loss_mask[selected_indices] = 1.0

    training_mask = input_mask.astype(np.float32) - loss_mask

    return training_mask, loss_mask


def nufft_sense_adjoint(
    operator,
    kspace,
    sample_mask,
    device,
):
    """
    Compute A^H M y for one non-Cartesian multi-coil acquisition.

    Parameters
    ----------
    operator
        MRI-NUFFT SENSE operator containing the full trajectory and
        sensitivity maps.
    kspace : ndarray, [n_coils, n_samples]
    sample_mask : ndarray, [n_samples]
    device : torch.device

    Returns
    -------
    image : complex64 ndarray
        [Nx, Ny] for 2D or [Nx, Ny, Nz] for 3D.
    """

    kspace = np.asarray(kspace)
    sample_mask = np.asarray(sample_mask)

    if kspace.ndim != 2:
        raise ValueError(
            f"Expected kspace [n_coils, n_samples], got {kspace.shape}"
        )

    if sample_mask.shape != (kspace.shape[-1],):
        raise ValueError(
            "sample_mask and kspace must have the same sample count: "
            f"{sample_mask.shape} versus {kspace.shape}"
        )

    kspace_torch = torch.as_tensor(
        kspace,
        dtype=torch.complex64,
        device=device,
    )

    mask_torch = torch.as_tensor(
        sample_mask,
        dtype=torch.float32,
        device=device,
    )

    masked_kspace = (
        kspace_torch * mask_torch.unsqueeze(0)
    ).unsqueeze(0)

    with torch.no_grad():
        image = operator.adj_op(masked_kspace)[0, 0]

    return image.detach().cpu().numpy().astype(np.complex64)


def getPSNR(reference, reconstruction):
    """
    Compute magnitude-image PSNR.
    """

    mse = np.mean(
        np.square(np.abs(reference - reconstruction))
    )

    peak = np.max(np.abs(reference))

    return 20 * np.log10(
        peak / (np.sqrt(mse) + 1e-10)
    )


def complex2real(input_data):
    """
    Convert a complex array [... dimensions ...] to [... dimensions ..., 2].
    """

    return np.stack(
        (input_data.real, input_data.imag),
        axis=-1,
    )


def real2complex(input_data):
    """
    Convert a real/imaginary array [... dimensions ..., 2] to complex.
    """

    return input_data[..., 0] + 1j * input_data[..., 1]