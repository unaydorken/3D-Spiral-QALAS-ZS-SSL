import sys

import h5py
import numpy as np
import twixtools


# ============================================================
# Siemens raw-data helper
# ============================================================

def get_image_mdbs(data_file):

    twix = twixtools.read_twix(
        data_file,
        parse_pmu=False,
    )

    meas = twix[-1]

    mdbs = []

    for mdb in meas["mdb"]:

        try:
            if not mdb.is_image_scan():
                continue
        except Exception:
            pass

        if (
            hasattr(mdb, "data")
            and len(mdb.data.shape) == 2
        ):
            mdbs.append(mdb)

    return mdbs


def is_reflected_mdb(mdb):

    try:
        return "REFLECT" in mdb.get_active_flags()
    except Exception as exc:
        raise RuntimeError(
            "Could not determine Siemens REFLECT flag "
            "for an MDB."
        ) from exc


def get_mdb_data(mdb):

    data = np.asarray(
        mdb.data,
        dtype=np.complex64,
    )

    if is_reflected_mdb(mdb):

        # Siemens MDH REFLECT means the readout direction
        # is reversed. Match the MATLAB/mapVBVD path:
        #
        # if refl(i)
        #     x = flip(x,1);
        # end
        #
        # mdb.data shape is [coil, sample], therefore flip
        # along the sample axis.
        data = data[:, ::-1].copy()

    return data


# ============================================================
# Cartesian reference extraction
# ============================================================

def extract_reference(
    data_file,
    ref_file,
    info_file,
    ref_samples,
    spiral_samples,
    expected_ref,
    expected_spiral,
):

    mdbs = get_image_mdbs(
        data_file
    )

    ref_mdbs = [
        m
        for m in mdbs
        if m.data.shape[-1] == ref_samples
    ]

    spiral_mdbs = [
        m
        for m in mdbs
        if m.data.shape[-1] == spiral_samples
    ]

    if len(ref_mdbs) != expected_ref:

        raise RuntimeError(
            f"Reference readouts: "
            f"{len(ref_mdbs)}, "
            f"expected {expected_ref}"
        )

    if len(spiral_mdbs) != expected_spiral:

        raise RuntimeError(
            f"Spiral readouts: "
            f"{len(spiral_mdbs)}, "
            f"expected {expected_spiral}"
        )

    ncoil = ref_mdbs[0].data.shape[0]

    reference = np.stack(
        [
            get_mdb_data(m).T
            for m in ref_mdbs
        ],
        axis=2,
    )

    reference.ravel(
        order="F"
    ).tofile(
        ref_file
    )

    np.savetxt(
        info_file,
        [
            ncoil,
            len(ref_mdbs),
            len(spiral_mdbs),
        ],
        fmt="%d",
    )

    print(
        "Physical coils:     ",
        ncoil,
    )

    print(
        "Reference readouts: ",
        len(ref_mdbs),
    )

    print(
        "Spiral readouts:    ",
        len(spiral_mdbs),
    )

    print(
        "Reference shape:    ",
        reference.shape,
    )


# ============================================================
# HDF5 generation
# ============================================================

def write_h5_3d_multicontrast(
    data_file,
    out_file,
    comp_file,
    traj_file,
    sens_file,
    ncoil,
    cc,
    ns,
    nz,
    ncontrast,
    ninterl_full,
    acceleration,
    nx,
    ny,
    slice_idx=None,
    motion_params_file=None,
    apply_moco=False,
):

    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    if acceleration < 1:

        raise RuntimeError(
            "Acceleration must be >= 1."
        )

    if ninterl_full % acceleration != 0:

        raise RuntimeError(
            "Number of interleaves is not "
            "divisible by acceleration."
        )

    ninterl_out = (
        ninterl_full
        // acceleration
    )

    # --------------------------------------------------------
    # Motion correction is only supported here in full 3D.
    #
    # After arbitrary 3D rotation, kz is mixed with kx/ky,
    # so the existing simple kz-IFFT slice extraction is no
    # longer valid.
    # --------------------------------------------------------

    if apply_moco and slice_idx is not None:

        raise RuntimeError(
            "Motion correction requires full 3D output. "
            "Set slice_idx='all'."
        )

    # --------------------------------------------------------
    # Retained R=1 raw interleaves
    #
    # C1/C3/C5:
    #     first ninterl_out arms
    #
    # C2/C4:
    #     last ninterl_out arms
    #
    # This matches the prospectively accelerated sequence.
    # --------------------------------------------------------

    kept_interleaves = np.empty(
        (
            ncontrast,
            ninterl_out,
        ),
        dtype=np.int64,
    )

    for c in range(ncontrast):

        if c in (0, 2, 4):

            kept_interleaves[
                c,
                :,
            ] = np.arange(
                0,
                ninterl_out,
                dtype=np.int64,
            )

        else:

            kept_interleaves[
                c,
                :,
            ] = np.arange(
                ninterl_full - ninterl_out,
                ninterl_full,
                dtype=np.int64,
            )

    print()
    print("Raw interleaves retained, 1-based:")

    for c in range(ncontrast):

        print(
            f"Contrast {c + 1}: "
            f"{kept_interleaves[c, 0] + 1} "
            f"... "
            f"{kept_interleaves[c, -1] + 1}"
        )

    # --------------------------------------------------------
    # Coil compression matrix
    # [physical coil, compressed coil]
    # --------------------------------------------------------

    C = np.fromfile(
        comp_file,
        dtype=np.complex64,
    ).reshape(
        (
            ncoil,
            cc,
        ),
        order="F",
    )

    # --------------------------------------------------------
    # FULL retained 3D trajectory
    #
    # [contrast, sample, xyz]
    #
    # Coordinates are cycles/voxel.
    #
    # sample ordering:
    # spiral sample -> kz -> retained interleave
    # --------------------------------------------------------

    n_samples_3d = (
        ns
        * nz
        * ninterl_out
    )

    trajectory_full = np.fromfile(
        traj_file,
        dtype=np.float32,
    ).reshape(
        (
            ncontrast,
            n_samples_3d,
            3,
        ),
        order="F",
    )

    # --------------------------------------------------------
    # FULL sensitivity maps
    # [Nx, Ny, Nz, compressed coil]
    # --------------------------------------------------------

    sens_maps_full = np.fromfile(
        sens_file,
        dtype=np.complex64,
    ).reshape(
        (
            nx,
            ny,
            nz,
            cc,
        ),
        order="F",
    )

    # --------------------------------------------------------
    # Optional motion parameters
    #
    # MATLAB writes:
    #
    # [12, ninterl_full]
    #
    # rows 1:9:
    #     R_array(:)
    #
    # rows 10:12:
    #     center-referenced translation t_eff
    #     in QALAS voxels
    #
    # Same convention as working MATLAB MoCo:
    #
    # k_cor = R_array.' * k_nom
    #
    # phase = exp(+i 2*pi*k_nom.'*t_eff)
    #
    # Python trajectory vectors are rows, so:
    #
    # k_cor_row = k_nom_row @ R_array
    # --------------------------------------------------------

    motion_params = None

    if apply_moco:

        if (
            motion_params_file is None
            or motion_params_file.lower() == "none"
        ):

            raise RuntimeError(
                "Motion correction requested but no "
                "motion parameter file was provided."
            )

        motion_params = np.fromfile(
            motion_params_file,
            dtype=np.float32,
        )

        expected_motion_values = (
            12
            * ninterl_full
        )

        if (
            motion_params.size
            != expected_motion_values
        ):

            raise RuntimeError(
                f"Motion parameter file has "
                f"{motion_params.size} values, "
                f"expected {expected_motion_values}."
            )

        motion_params = motion_params.reshape(
            (
                12,
                ninterl_full,
            ),
            order="F",
        )

        print()
        print(
            "Loaded motion parameters:",
            motion_params.shape,
        )

    # --------------------------------------------------------
    # Read full R=1 raw spiral data
    # --------------------------------------------------------

    mdbs = get_image_mdbs(
        data_file
    )

    spiral_mdbs = [
        m
        for m in mdbs
        if m.data.shape[-1] == ns
    ]

    n_reflected = sum(
        is_reflected_mdb(m)
        for m in spiral_mdbs
    )

    print()
    print(
        "Reflected QALAS readouts:",
        n_reflected,
        "/",
        len(spiral_mdbs),
    )

    expected = (
        nz
        * ncontrast
        * ninterl_full
    )

    if len(spiral_mdbs) != expected:

        raise RuntimeError(
            f"Spiral readouts: "
            f"{len(spiral_mdbs)}, "
            f"expected {expected}"
        )

    # ========================================================
    # FULL 3D MODE
    # ========================================================

    if slice_idx is None:

        n_samples_out = (
            ns
            * nz
            * ninterl_out
        )

        kspace = np.empty(
            (
                ncontrast,
                cc,
                n_samples_out,
            ),
            dtype=np.complex64,
        )

        trajectory = (
            trajectory_full.copy()
        )

        sens_maps = (
            sens_maps_full
        )

        # ----------------------------------------------------
        # Output retained-interleave index
        # ----------------------------------------------------

        for out_interl in range(
            ninterl_out
        ):

            start_out = (
                out_interl
                * ns
                * nz
            )

            stop_out = (
                start_out
                + ns * nz
            )

            # ------------------------------------------------
            # Each contrast can come from a DIFFERENT
            # original R=1 arm.
            # ------------------------------------------------

            for c in range(
                ncontrast
            ):

                raw_interl = int(
                    kept_interleaves[
                        c,
                        out_interl,
                    ]
                )

                first = (
                    raw_interl
                    * ncontrast
                    + c
                ) * nz

                raw = np.stack(
                    [
                        get_mdb_data(
                            spiral_mdbs[
                                first + z
                            ]
                        )
                        for z in range(nz)
                    ],
                    axis=0,
                )

                # [z, physical coil, sample]
                # ->
                # [z, compressed coil, sample]

                raw_cc = np.einsum(
                    "pc,zps->zcs",
                    C.conj(),
                    raw,
                    optimize=True,
                )

                # ->
                # [compressed coil, z*sample]

                block = (
                    raw_cc
                    .transpose(
                        1,
                        0,
                        2,
                    )
                    .reshape(
                        cc,
                        ns * nz,
                    )
                )

                # --------------------------------------------
                # Optional motion correction
                # --------------------------------------------

                if apply_moco:

                    # Use the pose of the ORIGINAL retained
                    # R=1 arm, not the output R=4 arm number.

                    p = motion_params[
                        :,
                        raw_interl,
                    ]

                    R_array = (
                        p[0:9]
                        .reshape(
                            (3, 3),
                            order="F",
                        )
                        .astype(
                            np.float64,
                            copy=False,
                        )
                    )

                    t_eff = (
                        p[9:12]
                        .astype(
                            np.float64,
                            copy=False,
                        )
                    )

                    # Nominal retained trajectory for this
                    # output block, cycles/voxel.

                    k_nom = (
                        trajectory_full[
                            c,
                            start_out:stop_out,
                            :,
                        ]
                        .astype(
                            np.float64,
                            copy=False,
                        )
                    )

                    # MATLAB column-vector convention:
                    #
                    # k_cor = R_array.' * k_nom
                    #
                    # Python row-vector equivalent:

                    k_cor = (
                        k_nom
                        @ R_array
                    )

                    # Translation phase correction.
                    # Use nominal k-space trajectory.

                    phase_corr = np.exp(
                        1j
                        * 2.0
                        * np.pi
                        * (
                            k_nom
                            @ t_eff
                        )
                    ).astype(
                        np.complex64
                    )

                    block = (
                        block
                        * phase_corr[
                            None,
                            :
                        ]
                    )

                    trajectory[
                        c,
                        start_out:stop_out,
                        :,
                    ] = k_cor.astype(
                        np.float32
                    )

                kspace[
                    c,
                    :,
                    start_out:stop_out,
                ] = block

        nz_out = nz

        slice_description = "all"

    # ========================================================
    # SINGLE ANATOMICAL SLICE MODE
    #
    # No MoCo only.
    # ========================================================

    else:

        if not (
            0 <= slice_idx < nz
        ):

            raise RuntimeError(
                f"slice_idx={slice_idx} "
                f"is outside 0..{nz - 1}"
            )

        n_samples_out = (
            ns
            * ninterl_out
        )

        nz_out = 1

        kspace = np.empty(
            (
                ncontrast,
                cc,
                n_samples_out,
            ),
            dtype=np.complex64,
        )

        trajectory = np.zeros(
            (
                ncontrast,
                n_samples_out,
                2,
            ),
            dtype=np.float32,
        )

        sens_maps = (
            sens_maps_full[
                :,
                :,
                slice_idx:slice_idx + 1,
                :,
            ]
        )

        xy_tol = 2e-5
        kz_tol = 2e-5
        spacing_tol = 2e-5

        for out_interl in range(
            ninterl_out
        ):

            start_3d = (
                out_interl
                * ns
                * nz
            )

            stop_3d = (
                start_3d
                + ns * nz
            )

            start_out = (
                out_interl
                * ns
            )

            stop_out = (
                start_out
                + ns
            )

            for c in range(
                ncontrast
            ):

                raw_interl = int(
                    kept_interleaves[
                        c,
                        out_interl,
                    ]
                )

                first = (
                    raw_interl
                    * ncontrast
                    + c
                ) * nz

                raw = np.stack(
                    [
                        get_mdb_data(
                            spiral_mdbs[
                                first + z
                            ]
                        )
                        for z in range(nz)
                    ],
                    axis=0,
                )

                raw_cc = np.einsum(
                    "pc,zps->zcs",
                    C.conj(),
                    raw,
                    optimize=True,
                )

                traj_block = (
                    trajectory_full[
                        c,
                        start_3d:stop_3d,
                        :,
                    ]
                    .reshape(
                        nz,
                        ns,
                        3,
                    )
                )

                # --------------------------------------------
                # Check kz constant during each spiral readout
                # --------------------------------------------

                kz_per_encode = np.mean(
                    traj_block[:, :, 2],
                    axis=1,
                )

                kz_variation = np.max(
                    np.abs(
                        traj_block[:, :, 2]
                        - kz_per_encode[:, None]
                    )
                )

                if kz_variation > kz_tol:

                    raise RuntimeError(
                        "kz changes during the spiral "
                        f"readout. Max change = "
                        f"{kz_variation:.3e}."
                    )

                # --------------------------------------------
                # Check kx/ky identical across kz
                # --------------------------------------------

                center_kz = int(
                    np.argmin(
                        np.abs(
                            kz_per_encode
                        )
                    )
                )

                xy_reference = (
                    traj_block[
                        center_kz,
                        :,
                        :2,
                    ]
                )

                xy_error = np.max(
                    np.abs(
                        traj_block[:, :, :2]
                        - xy_reference[
                            None,
                            :,
                            :,
                        ]
                    )
                )

                if xy_error > xy_tol:

                    raise RuntimeError(
                        "kx/ky changes across kz. "
                        f"Max difference = "
                        f"{xy_error:.3e}."
                    )

                # --------------------------------------------
                # Sort by actual kz
                # --------------------------------------------

                kz_order = np.argsort(
                    kz_per_encode
                )

                kz_sorted = (
                    kz_per_encode[
                        kz_order
                    ]
                )

                raw_sorted = (
                    raw_cc[
                        kz_order,
                        :,
                        :,
                    ]
                )

                # --------------------------------------------
                # Verify uniform Cartesian kz spacing
                # --------------------------------------------

                dk = np.diff(
                    kz_sorted
                )

                expected_dk = (
                    1.0
                    / float(nz)
                )

                spacing_error = np.max(
                    np.abs(
                        dk
                        - expected_dk
                    )
                )

                if spacing_error > spacing_tol:

                    raise RuntimeError(
                        "kz encodes are not uniformly "
                        "Cartesian. "
                        f"Max spacing error = "
                        f"{spacing_error:.3e}."
                    )

                # --------------------------------------------
                # kz -> anatomical z
                # --------------------------------------------

                raw_z = np.fft.fftshift(
                    np.fft.ifft(
                        np.fft.ifftshift(
                            raw_sorted,
                            axes=0,
                        ),
                        axis=0,
                    ),
                    axes=0,
                )

                raw_slice = np.asarray(
                    raw_z[
                        slice_idx,
                        :,
                        :,
                    ],
                    dtype=np.complex64,
                )

                kspace[
                    c,
                    :,
                    start_out:stop_out,
                ] = raw_slice

                trajectory[
                    c,
                    start_out:stop_out,
                    0:2,
                ] = xy_reference

        slice_description = (
            f"anatomical slice "
            f"{slice_idx}"
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    print()

    print(
        "Raw interleaves:    ",
        ninterl_full,
    )

    print(
        "Acceleration:       ",
        acceleration,
    )

    print(
        "Output interleaves: ",
        ninterl_out,
    )

    print(
        "Slice selection:    ",
        slice_description,
    )

    print(
        "Motion correction:  ",
        apply_moco,
    )

    print()

    print("Final shapes:")

    print(
        "kspace:     ",
        kspace.shape,
    )

    print(
        "trajectory: ",
        trajectory.shape,
    )

    print(
        "sens_maps:  ",
        sens_maps.shape,
    )

    print()

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert kspace.shape == (
        ncontrast,
        cc,
        n_samples_out,
    )

    expected_nspatial = (
        3
        if slice_idx is None
        else 2
    )

    assert trajectory.shape == (
        ncontrast,
        n_samples_out,
        expected_nspatial,
    )

    assert sens_maps.shape == (
        nx,
        ny,
        nz_out,
        cc,
    )

    if not np.all(
        np.isfinite(kspace)
    ):

        raise RuntimeError(
            "Non-finite values found in k-space."
        )

    if not np.all(
        np.isfinite(trajectory)
    ):

        raise RuntimeError(
            "Non-finite values found in trajectory."
        )

    if not np.all(
        np.isfinite(sens_maps)
    ):

        raise RuntimeError(
            "Non-finite values found in sensitivity maps."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with h5py.File(
        out_file,
        "w",
    ) as f:

        f.create_dataset(
            "kspace",
            data=kspace,
        )

        f.create_dataset(
            "trajectory",
            data=trajectory,
        )

        f.create_dataset(
            "sens_maps",
            data=sens_maps,
        )

    print(
        "Saved:",
        out_file,
    )



# ============================================================
# Write HDF5 from arrays already prepared in MATLAB
# ============================================================

def write_h5_precomputed(
    out_file,
    kspace_file,
    traj_file,
    sens_file,
    ncontrast,
    cc,
    nsamples,
    nx,
    ny,
    nz,
):
    """
    Package arrays already prepared in MATLAB.

    MATLAB writes column-major binaries with these logical shapes:

        kspace     [ncontrast, cc, nsamples] complex64
        trajectory [ncontrast, nsamples, 3]  float32
        sens_maps  [nx, ny, nz, cc]          complex64

    No raw-data parsing, coil compression, trajectory correction,
    or motion correction is performed here.
    """

    kspace = np.fromfile(
        kspace_file,
        dtype=np.complex64,
    ).reshape(
        (
            ncontrast,
            cc,
            nsamples,
        ),
        order="F",
    )

    trajectory = np.fromfile(
        traj_file,
        dtype=np.float32,
    ).reshape(
        (
            ncontrast,
            nsamples,
            3,
        ),
        order="F",
    )

    sens_maps = np.fromfile(
        sens_file,
        dtype=np.complex64,
    ).reshape(
        (
            nx,
            ny,
            nz,
            cc,
        ),
        order="F",
    )

    if not np.all(np.isfinite(kspace)):
        raise RuntimeError(
            "Non-finite values found in precomputed k-space."
        )

    if not np.all(np.isfinite(trajectory)):
        raise RuntimeError(
            "Non-finite values found in precomputed trajectory."
        )

    if not np.all(np.isfinite(sens_maps)):
        raise RuntimeError(
            "Non-finite values found in precomputed sensitivity maps."
        )

    print()
    print("Precomputed arrays:")
    print("kspace:     ", kspace.shape)
    print("trajectory: ", trajectory.shape)
    print("sens_maps:  ", sens_maps.shape)
    print(
        "k-space norm:",
        f"{np.linalg.norm(kspace.ravel()):.8e}",
    )

    for d in range(3):
        v = trajectory[..., d]
        print(
            f"trajectory dim {d + 1}: "
            f"[{v.min():+.6f}, {v.max():+.6f}]"
        )

    with h5py.File(
        out_file,
        "w",
    ) as f:

        f.create_dataset(
            "kspace",
            data=kspace,
        )

        f.create_dataset(
            "trajectory",
            data=trajectory,
        )

        f.create_dataset(
            "sens_maps",
            data=sens_maps,
        )

    print()
    print("Saved:", out_file)


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        raise RuntimeError(
            "A mode must be specified."
        )

    mode = sys.argv[1]

    # --------------------------------------------------------
    # Reference extraction
    # --------------------------------------------------------

    if mode == "extract_reference":

        if len(sys.argv) != 9:

            raise RuntimeError(
                "extract_reference expects "
                "7 arguments."
            )

        extract_reference(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
            int(sys.argv[6]),
            int(sys.argv[7]),
            int(sys.argv[8]),
        )

    # --------------------------------------------------------
    # Package arrays already prepared in MATLAB
    # --------------------------------------------------------

    elif mode == "write_h5_precomputed":

        # Arguments after mode:
        #
        #  1  out_file
        #  2  kspace_file
        #  3  traj_file
        #  4  sens_file
        #  5  ncontrast
        #  6  cc
        #  7  nsamples
        #  8  nx
        #  9  ny
        # 10  nz

        if len(sys.argv) != 12:

            raise RuntimeError(
                "write_h5_precomputed expects "
                "10 arguments after mode."
            )

        write_h5_precomputed(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
            int(sys.argv[6]),
            int(sys.argv[7]),
            int(sys.argv[8]),
            int(sys.argv[9]),
            int(sys.argv[10]),
            int(sys.argv[11]),
        )

    # --------------------------------------------------------
    # HDF5 generation from raw TWIX
    # --------------------------------------------------------

    elif mode == "write_h5_3d_multicontrast":

        # Arguments after mode:
        #
        #  1  data_file
        #  2  out_file
        #  3  comp_file
        #  4  traj_file
        #  5  sens_file
        #  6  ncoil
        #  7  cc
        #  8  ns
        #  9  nz
        # 10  ncontrast
        # 11  ninterl_full
        # 12  acceleration
        # 13  nx
        # 14  ny
        # 15  slice_arg
        # 16  motion_params_file
        # 17  apply_moco

        if len(sys.argv) != 19:

            raise RuntimeError(
                "write_h5_3d_multicontrast "
                "expects 17 arguments after mode."
            )

        if sys.argv[16] == "all":

            slice_idx = None

        else:

            slice_idx = int(
                sys.argv[16]
            )

        motion_params_file = (
            sys.argv[17]
        )

        apply_moco = bool(
            int(
                sys.argv[18]
            )
        )

        write_h5_3d_multicontrast(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
            sys.argv[6],
            int(sys.argv[7]),
            int(sys.argv[8]),
            int(sys.argv[9]),
            int(sys.argv[10]),
            int(sys.argv[11]),
            int(sys.argv[12]),
            int(sys.argv[13]),
            int(sys.argv[14]),
            int(sys.argv[15]),
            slice_idx=slice_idx,
            motion_params_file=motion_params_file,
            apply_moco=apply_moco,
        )

    else:

        raise RuntimeError(
            f"Unknown mode: {mode}"
        )