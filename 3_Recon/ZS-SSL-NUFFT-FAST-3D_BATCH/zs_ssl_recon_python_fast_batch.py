import gc
import os
import time

import cupy as cp
import h5py
import numpy as np
import scipy.io as sio
import torch

from mrinufft import get_operator
from mrinufft.operators import compute_toeplitz_kernel
from torch.utils.data import DataLoader

from models import parser_ops, UnrollNet_fast_batch as UnrollNet, utils
from models.modules import (
    Dataset,
    MixL1L2Loss,
    train,
    validation,
)


# ========================================================================
# MRI-NUFFT trajectory conversion
# ========================================================================

def trajectory_to_mrinufft(trajectory):

    trajectory = np.asarray(
        trajectory,
        dtype=np.float32,
    )

    # H5 convention: cycles / voxel.
    #
    # Rigid rotation can push individual coordinates slightly outside
    # [-0.5, 0.5). Wrap equivalent discrete Fourier frequencies back
    # into the principal interval.
    trajectory = (
        (
            trajectory
            + np.float32(0.5)
        )
        % np.float32(1.0)
        - np.float32(0.5)
    )

    # cuFINUFFT convention used internally by MRI-NUFFT: radians.
    # Supplying radians explicitly prevents proper_trajectory() from
    # guessing units from max(abs(trajectory)).
    trajectory = (
        trajectory
        * np.float32(2.0 * np.pi)
    )

    return np.ascontiguousarray(
        trajectory,
        dtype=np.float32,
    )


# ========================================================================
# Main
# ========================================================================

def main():

    # --------------------------------------------------------------------
    # Runtime configuration
    # --------------------------------------------------------------------

    backend = os.environ.get(
        "MRINUFFT_BACKEND",
        "cufinufft",
    )

    if backend == "cufinufft":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "MRINUFFT_BACKEND=cufinufft was requested, "
                "but PyTorch cannot access a CUDA GPU."
            )

        device = torch.device("cuda")

    elif backend == "finufft":

        device = torch.device("cpu")

    else:

        raise ValueError(
            f"Expected backend 'finufft' or 'cufinufft', "
            f"got {backend!r}"
        )

    parser = parser_ops.get_parser()
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("--data_dir is required")

    print(
        "Device:",
        device,
        flush=True,
    )

    print(
        "MRI-NUFFT backend:",
        backend,
        flush=True,
    )

    if device.type == "cuda":

        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    # --------------------------------------------------------------------
    # FAST requires cuFINUFFT
    # --------------------------------------------------------------------

    if backend != "cufinufft":

        raise RuntimeError(
            "ZS-SSL-NUFFT-FAST Toeplitz mode currently requires "
            "MRINUFFT_BACKEND=cufinufft."
        )

    # --------------------------------------------------------------------
    # Load HDF5
    # --------------------------------------------------------------------

    print(
        f"Loading data from {args.data_dir}",
        flush=True,
    )

    with h5py.File(
        args.data_dir,
        "r",
    ) as f:

        kspace = f[
            "kspace"
        ][()].astype(
            np.complex64
        )

        trajectory = f[
            "trajectory"
        ][()].astype(
            np.float32
        )

        sens_maps = f[
            "sens_maps"
        ][()].astype(
            np.complex64
        )

    # --------------------------------------------------------------------
    # Canonical dimensions
    #
    # kspace:
    #     [contrast, coil, sample]
    #
    # trajectory:
    #     [contrast, sample, spatial]
    #
    # sens_maps:
    #     [Nx, Ny, Nz, coil]
    # --------------------------------------------------------------------

    # Old single-contrast files:
    # [coil, sample] -> [1, coil, sample]

    if kspace.ndim == 2:
        kspace = kspace[None, ...]

    # Old trajectory:
    # [sample, spatial] -> [1, sample, spatial]

    if trajectory.ndim == 2:
        trajectory = trajectory[None, ...]

    # Old 2D sensitivity maps:
    # [Nx, Ny, coil] -> [Nx, Ny, 1, coil]

    if sens_maps.ndim == 3:

        sens_maps = sens_maps[
            :,
            :,
            None,
            :,
        ]

    ncontrast, ncoil, nsamples = (
        kspace.shape
    )

    nspatial = (
        trajectory.shape[-1]
    )

    nx, ny, nz, ncoil_sens = (
        sens_maps.shape
    )

    # --------------------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------------------

    assert (
        trajectory.shape[0]
        == ncontrast
    )

    assert (
        trajectory.shape[1]
        == nsamples
    )

    assert (
        ncoil_sens
        == ncoil
    )

    # --------------------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------------------

    if kspace.ndim != 3:

        raise ValueError(
            "Expected kspace "
            "[contrast, coil, sample], "
            f"got {kspace.shape}"
        )

    if trajectory.ndim != 3:

        raise ValueError(
            "Expected trajectory "
            "[contrast, sample, spatial], "
            f"got {trajectory.shape}"
        )

    if sens_maps.ndim != 4:

        raise ValueError(
            "Expected sens_maps "
            "[Nx, Ny, Nz, coil], "
            f"got {sens_maps.shape}"
        )

    if nspatial not in (
        2,
        3,
    ):

        raise ValueError(
            f"Expected 2D or 3D trajectory, "
            f"got {nspatial}D"
        )

    if trajectory.shape != (
        ncontrast,
        nsamples,
        nspatial,
    ):

        raise ValueError(
            f"Invalid trajectory shape: "
            f"{trajectory.shape}"
        )

    if sens_maps.shape[-1] != ncoil:

        raise ValueError(
            "Sensitivity-map coil count "
            "does not match k-space."
        )

    # --------------------------------------------------------------------
    # 2D / 3D reconstruction configuration
    # --------------------------------------------------------------------

    if nspatial == 3:

        print(
            "3D configuration:",
            {
                "batchSize": args.batchSize,
                "num_reps": args.num_reps,
                "epochs": args.epochs,
                "unrolls": args.nb_unroll_blocks,
                "CG_max_iter": args.CG_Iter,
                "CG_tol": args.CG_tol,
            },
            flush=True,
        )

    else:

        print(
            "2D configuration:",
            {
                "batchSize": args.batchSize,
                "num_reps": args.num_reps,
                "epochs": args.epochs,
                "unrolls": args.nb_unroll_blocks,
                "CG_max_iter": args.CG_Iter,
                "CG_tol": args.CG_tol,
            },
            flush=True,
        )

    # --------------------------------------------------------------------
    # Global k-space normalization
    # --------------------------------------------------------------------

    kspace_scale = np.max(
        np.abs(kspace)
    )

    if (
        not np.isfinite(kspace_scale)
        or kspace_scale == 0
    ):

        raise ValueError(
            f"Invalid k-space normalization scale: "
            f"{kspace_scale}"
        )

    kspace = (
        kspace
        / kspace_scale
    ).astype(
        np.complex64
    )

    # --------------------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------------------

    if nspatial == 2:

        if nz != 1:

            raise ValueError(
                "2D reconstruction expects singleton z dimension, "
                f"got sens_maps {sens_maps.shape}"
            )

        image_shape = (
            nx,
            ny,
        )

        # [Nx, Ny, 1, coil]
        # ->
        # [coil, Nx, Ny]

        sens_maps_operator = np.ascontiguousarray(
            np.moveaxis(
                sens_maps[
                    :,
                    :,
                    0,
                    :,
                ],
                -1,
                0,
            ),
            dtype=np.complex64,
        )

    else:

        image_shape = (
            nx,
            ny,
            nz,
        )

        # [Nx, Ny, Nz, coil]
        # ->
        # [coil, Nx, Ny, Nz]

        sens_maps_operator = np.ascontiguousarray(
            np.moveaxis(
                sens_maps,
                -1,
                0,
            ),
            dtype=np.complex64,
        )

    n_coils = ncoil
    n_samples = nsamples

    # Existing dimensions

    args.nrow_GLOB = nx
    args.ncol_GLOB = ny
    args.ncoil_GLOB = n_coils
    args.ncontrast_GLOB = ncontrast

    # Additional 3D information

    args.nz_GLOB = nz
    args.nspatial_GLOB = nspatial

    # --------------------------------------------------------------------
    # Coil batching
    # --------------------------------------------------------------------

    if nspatial == 2:

        n_trans_fast = n_coils
        smaps_cached_fast = True

    else:

        n_trans_fast = 1
        smaps_cached_fast = False

    # --------------------------------------------------------------------
    # Build operators
    # --------------------------------------------------------------------

    operators = []
    toeplitz_operators = []

    for c in range(
        ncontrast
    ):

        trajectory_c = trajectory_to_mrinufft(
            trajectory[c]
        )

        trajectory_torch_c = (
            torch.from_numpy(
                trajectory_c
            ).to(
                device
            )
        )

        # ---------------------------------------------------------------
        # Standard differentiable SENSE NUFFT
        # ---------------------------------------------------------------

        operator_c = get_operator(
            backend,
            wrt_data=True,
        )(
            samples=trajectory_torch_c,
            shape=image_shape,
            n_coils=n_coils,
            smaps=sens_maps_operator,
            density=False,
            squeeze_dims=False,
            smaps_cached=smaps_cached_fast,
            n_trans=n_trans_fast,
        )

        operators.append(
            operator_c
        )

        # ---------------------------------------------------------------
        # Toeplitz SENSE operator
        # ---------------------------------------------------------------

        toeplitz_operator_c = get_operator(
            "cufinufft",
            samples=trajectory_c,
            shape=image_shape,
            n_coils=n_coils,
            smaps=sens_maps_operator,
            density=False,
            squeeze_dims=True,
            smaps_cached=smaps_cached_fast,
            n_trans=n_trans_fast,
        )

        toeplitz_operators.append(
            toeplitz_operator_c
        )

    # --------------------------------------------------------------------
    # Fixed validation partition
    # --------------------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    original_mask = np.ones(
        (
            ncontrast,
            n_samples,
        ),
        dtype=np.float32,
    )

    cv_trn_mask = np.empty_like(
        original_mask
    )

    cv_val_mask = np.empty_like(
        original_mask
    )

    for c in range(
        ncontrast
    ):

        (
            cv_trn_mask[c],
            cv_val_mask[c],
        ) = utils.split_sample_mask(
            trajectory=trajectory[c],
            input_mask=original_mask[c],
            rho=args.rho_val,
            center_radius_fraction=0.05,
            rng=rng,
        )

    remainder_mask = (
        cv_trn_mask.copy()
    )

    # --------------------------------------------------------------------
    # Validation network input
    # --------------------------------------------------------------------

    nw_input_val = np.empty(
        (
            1,
            ncontrast,
            *image_shape,
        ),
        dtype=np.complex64,
    )

    for c in range(
        ncontrast
    ):

        nw_input_val[
            0,
            c,
        ] = utils.nufft_sense_adjoint(
            operator=operators[c],
            kspace=kspace[c],
            sample_mask=cv_trn_mask[c],
            device=device,
        )

    # --------------------------------------------------------------------
    # Validate validation partition
    # --------------------------------------------------------------------

    assert np.all(
        cv_trn_mask
        + cv_val_mask
        == original_mask
    )

    assert not np.any(
        (cv_trn_mask != 0)
        & (cv_val_mask != 0)
    )

    # --------------------------------------------------------------------
    # Repeated SSDU training partitions
    # --------------------------------------------------------------------

    nw_input_trn = np.empty(
        (
            args.num_reps,
            ncontrast,
            *image_shape,
        ),
        dtype=np.complex64,
    )

    trn_mask = np.empty(
        (
            args.num_reps,
            ncontrast,
            n_samples,
        ),
        dtype=np.float32,
    )

    loss_mask = np.empty(
        (
            args.num_reps,
            ncontrast,
            n_samples,
        ),
        dtype=np.float32,
    )

    for repetition in range(
        args.num_reps
    ):

        for c in range(
            ncontrast
        ):

            (
                trn_mask[
                    repetition,
                    c,
                ],
                loss_mask[
                    repetition,
                    c,
                ],
            ) = utils.split_sample_mask(
                trajectory=trajectory[c],
                input_mask=remainder_mask[c],
                rho=args.rho_train,
                center_radius_fraction=0.05,
                rng=rng,
            )

            # -----------------------------------------------------------
            # A_c^H M_Theta,c y_c
            # -----------------------------------------------------------

            nw_input_trn[
                repetition,
                c,
            ] = utils.nufft_sense_adjoint(
                operator=operators[c],
                kspace=kspace[c],
                sample_mask=trn_mask[
                    repetition,
                    c,
                ],
                device=device,
            )

    # --------------------------------------------------------------------
    # Validate training partitions
    # --------------------------------------------------------------------

    for repetition in range(
        args.num_reps
    ):

        assert np.all(
            trn_mask[
                repetition
            ]
            + loss_mask[
                repetition
            ]
            == remainder_mask
        )

        assert not np.any(
            (
                trn_mask[
                    repetition
                ]
                != 0
            )
            & (
                loss_mask[
                    repetition
                ]
                != 0
            )
        )

    # ====================================================================
    # Toeplitz kernel generation
    # ====================================================================

    n_kernel_sets = (
        args.num_reps
        + 1
    )

    toeplitz_kernels = [
        [
            None
            for _ in range(
                ncontrast
            )
        ]
        for _ in range(
            n_kernel_sets
        )
    ]

    # --------------------------------------------------------------------
    # Work contrast-by-contrast so only one temporary kernel operator
    # exists at a time.
    # --------------------------------------------------------------------

    for c in range(
        ncontrast
    ):

        trajectory_c = trajectory_to_mrinufft(
            trajectory[c]
        )

        kernel_operator_c = get_operator(
            "cufinufft",
            samples=trajectory_c,
            shape=image_shape,
            n_coils=1,
            density=False,
            squeeze_dims=True,
            n_trans=1,
        )

        # ---------------------------------------------------------------
        # Training kernels
        # ---------------------------------------------------------------

        for repetition in range(
            args.num_reps
        ):

            mask_weights = cp.asarray(
                trn_mask[
                    repetition,
                    c,
                ],
                dtype=cp.complex64,
            )

            kernel = compute_toeplitz_kernel(
                kernel_operator_c,
                weights=mask_weights,
            )

            # Store the kernel on CPU, not GPU.
            toeplitz_kernels[
                repetition
            ][
                c
            ] = cp.asnumpy(kernel)

            del mask_weights
            del kernel

        # ---------------------------------------------------------------
        # Validation kernel
        # ---------------------------------------------------------------

        mask_weights = cp.asarray(
            cv_trn_mask[c],
            dtype=cp.complex64,
        )

        kernel = compute_toeplitz_kernel(
            kernel_operator_c,
            weights=mask_weights,
        )

        toeplitz_kernels[
            args.num_reps
        ][
            c
        ] = cp.asnumpy(kernel)

        del mask_weights
        del kernel

        # ---------------------------------------------------------------
        # Temporary kernel-generation operator no longer needed
        # ---------------------------------------------------------------

        del kernel_operator_c

        gc.collect()

        torch.cuda.empty_cache()

        cp.get_default_memory_pool().free_all_blocks()

        cp.get_default_pinned_memory_pool().free_all_blocks()

    # ====================================================================
    # Convert complex arrays to real / imaginary channels
    # ====================================================================

    nw_input_trn_real = (
        utils.complex2real(
            nw_input_trn
        ).astype(
            np.float32
        )
    )

    # Full normalized k-space stored ONCE.
    #
    # [contrast, coil, sample]
    # ->
    # [contrast, coil, sample, 2]
    ref_kspace_real = (
        utils.complex2real(
            kspace
        ).astype(
            np.float32
        )
    )

    nw_input_val_real = (
        utils.complex2real(
            nw_input_val
        ).astype(
            np.float32
        )
    )

    # --------------------------------------------------------------------
    # Complex CPU arrays are no longer needed after conversion.
    # --------------------------------------------------------------------

    del nw_input_trn
    del nw_input_val

    # ref_kspace_real now contains the normalized k-space,
    # so the original complex array is no longer needed.
    del kspace

    gc.collect()

    # ====================================================================
    # Stable Toeplitz kernel IDs
    # ====================================================================

    train_kernel_indices = np.arange(
        args.num_reps,
        dtype=np.int64,
    )

    val_kernel_indices = np.array(
        [
            args.num_reps
        ],
        dtype=np.int64,
    )

    # ====================================================================
    # Data loaders
    # ====================================================================

    train_data = Dataset(
        nw_input_trn_real,
        trn_mask,
        loss_mask,
        ref_kspace_real,
        kernel_indices=(
            train_kernel_indices
        ),
    )

    pin_memory = (
        device.type == "cuda"
        and nspatial == 2
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_data = Dataset(
        nw_input_val_real,
        cv_trn_mask[
            np.newaxis,
            :
        ],
        cv_val_mask[
            np.newaxis,
            :
        ],
        ref_kspace_real,
        kernel_indices=(
            val_kernel_indices
        ),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # ====================================================================
    # Verify one batch
    # ====================================================================

    train_batch = next(
        iter(train_loader)
    )

    val_batch = next(
        iter(val_loader)
    )

    assert (
        len(train_batch)
        == 5
    )

    assert train_batch[
        0
    ].shape[
        1:
    ] == (
        ncontrast,
        *image_shape,
        2,
    )

    assert train_batch[
        1
    ].shape[
        1:
    ] == (
        ncontrast,
        n_samples,
    )

    assert train_batch[
        2
    ].shape[
        1:
    ] == (
        ncontrast,
        n_samples,
    )

    assert train_batch[
        3
    ].shape[
        1:
    ] == (
        ncontrast,
        n_coils,
        n_samples,
        2,
    )

    assert (
        train_batch[4].dtype
        == torch.int64
    )

    del train_batch
    del val_batch

    gc.collect()

    # ====================================================================
    # Output directory
    # ====================================================================

    directory = os.path.join(
        "saved_models",
        args.saved_model_name,
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    best_checkpoint_file = (
        os.path.join(
            directory,
            "best.pth",
        )
    )

    last_checkpoint_file = (
        os.path.join(
            directory,
            "last.pth",
        )
    )

    log_file = os.path.join(
        directory,
        "TrainingLog.mat",
    )

    # ====================================================================
    # Network
    # ====================================================================

    model = UnrollNet.UnrolledNet(
        args=args,
        operators=operators,
        toeplitz_operators=(
            toeplitz_operators
        ),
        toeplitz_kernels=(
            toeplitz_kernels
        ),
        device=device,
        ndim=nspatial,
    ).to(
        device
    )

    loss_fn = MixL1L2Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Cosine annealing schedule: lr decays from learning_rate → lr_min
    # over args.epochs steps.  Disabled when lr_min=0 (default) since
    # CosineAnnealingLR with eta_min=0 still decays — we use a flat
    # schedule flag instead to preserve backward compatibility.
    use_schedule = getattr(args, 'lr_min', 0.0) > 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=getattr(args, 'lr_min', 0.0),
    ) if use_schedule else None

    total_train_loss = []
    total_val_loss = []

    valid_loss_min = np.inf
    val_loss_tracker = 0
    start_epoch = 0

    # ====================================================================
    # Resume
    # ====================================================================

    if os.path.isfile(
        last_checkpoint_file
    ):

        checkpoint = torch.load(
            last_checkpoint_file,
            map_location=device,
        )

        checkpoint_nspatial = (
            checkpoint.get(
                "nspatial",
                nspatial,
            )
        )

        if (
            int(checkpoint_nspatial)
            != int(nspatial)
        ):

            raise RuntimeError(
                "Checkpoint dimensionality does not match "
                f"current reconstruction: "
                f"checkpoint={checkpoint_nspatial}D, "
                f"current={nspatial}D"
            )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optim_state"
            ]
        )

        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state"]
            )

        start_epoch = (
            checkpoint[
                "epoch"
            ]
            + 1
        )

        valid_loss_min = checkpoint[
            "valid_loss_min"
        ]

        val_loss_tracker = checkpoint[
            "val_loss_tracker"
        ]

        total_train_loss = checkpoint[
            "train_loss_history"
        ]

        total_val_loss = checkpoint[
            "val_loss_history"
        ]

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}",
            flush=True,
        )

    print(
        f"Training for at most "
        f"{args.epochs} epochs; "
        f"saving to {directory}",
        flush=True,
    )

    # ====================================================================
    # Epoch loop
    # ====================================================================

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        epoch_start = (
            time.time()
        )

        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------

        train_loss, learned_lambda = train(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device=device,
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cp.get_default_memory_pool().free_all_blocks()

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------

        val_loss = validation(
            val_loader,
            model,
            loss_fn,
            device=device,
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cp.get_default_memory_pool().free_all_blocks()

        # ---------------------------------------------------------------
        # LR schedule step
        # ---------------------------------------------------------------

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.learning_rate

        # ---------------------------------------------------------------
        # Numerical checks
        # ---------------------------------------------------------------

        if not np.isfinite(
            train_loss
        ):

            raise RuntimeError(
                f"Non-finite training loss "
                f"at epoch {epoch + 1}"
            )

        if not np.isfinite(
            val_loss
        ):

            raise RuntimeError(
                f"Non-finite validation loss "
                f"at epoch {epoch + 1}"
            )

        total_train_loss.append(
            train_loss
        )

        total_val_loss.append(
            val_loss
        )

        improved = (
            val_loss
            <= valid_loss_min
        )

        if improved:

            valid_loss_min = (
                val_loss
            )

            val_loss_tracker = 0

        else:

            val_loss_tracker += 1

        # ================================================================
        # Checkpoint
        # ================================================================

        checkpoint = {

            "epoch": epoch,

            "valid_loss_min": (
                valid_loss_min
            ),

            "val_loss_tracker": (
                val_loss_tracker
            ),

            "model_state": (
                model.state_dict()
            ),

            "optim_state": (
                optimizer.state_dict()
            ),

            "scheduler_state": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "train_loss_history": (
                total_train_loss
            ),

            "val_loss_history": (
                total_val_loss
            ),

            "backend": backend,

            "image_shape": (
                image_shape
            ),

            "nspatial": (
                nspatial
            ),

            "n_coils": (
                n_coils
            ),

            "n_samples": (
                n_samples
            ),

            "ncontrast": (
                ncontrast
            ),

            "kspace_scale": float(
                kspace_scale
            ),

            "nb_unroll_blocks": (
                args.nb_unroll_blocks
            ),

            "CG_Iter": (
                args.CG_Iter
            ),

            "CG_tol": (
                args.CG_tol
            ),

            "toeplitz": True,

            "n_trans": (
                n_trans_fast
            ),
        }

        # ---------------------------------------------------------------
        # Save newest state
        # ---------------------------------------------------------------

        torch.save(
            checkpoint,
            last_checkpoint_file,
        )

        # ---------------------------------------------------------------
        # Save best state
        # ---------------------------------------------------------------

        if improved:

            torch.save(
                checkpoint,
                best_checkpoint_file,
            )

        # ---------------------------------------------------------------
        # Training log
        # ---------------------------------------------------------------

        sio.savemat(
            log_file,
            {
                "trn_loss": np.asarray(
                    total_train_loss
                ),
                "val_loss": np.asarray(
                    total_val_loss
                ),
            },
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{args.epochs}, "
            f"time={elapsed:.2f}s, "
            f"train={train_loss:.6f}, "
            f"validation={val_loss:.6f}, "
            f"best={valid_loss_min:.6f}, "
            f"lambda={learned_lambda.item():.6f}, "
            f"lr={current_lr:.2e}, "
            f"patience="
            f"{val_loss_tracker}/"
            f"{args.stop_training}",
            flush=True,
        )

        # ---------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------

        if (
            val_loss_tracker
            >= args.stop_training
        ):

            print(
                f"Early stopping after "
                f"epoch {epoch + 1}.",
                flush=True,
            )

            break

    # ====================================================================
    # Final checks
    # ====================================================================

    if not os.path.isfile(
        best_checkpoint_file
    ):

        raise RuntimeError(
            f"Best checkpoint was not saved: "
            f"{best_checkpoint_file}"
        )

    print(
        "Training completed. "
        f"Best checkpoint: "
        f"{best_checkpoint_file}",
        flush=True,
    )


# ========================================================================
# Entry point
# ========================================================================

if __name__ == "__main__":
    main()