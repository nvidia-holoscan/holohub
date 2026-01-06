"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import numpy as np
from numpy.typing import NDArray

# reexport
from .hbo import ExtinctionCoefficient, HbO 
from .types import ChannelHeadsetMapping

logger = logging.getLogger(__name__)
USE_GPU_DEFAULT = True


@dataclass(frozen=True)
class Assets:
    mega_jacobian: NDArray[np.float32]  # float32 array (feature-channels, voxels)
    channel_mapping: ChannelHeadsetMapping  # mapping of channels to jacobian channel indices
    mua: NDArray[np.float32]  # absorption coefficients
    musp: NDArray[np.float32]  # reduced scattering coefficients
    idxs_significant_voxels: NDArray[np.int_]  # indices of significant voxels in mega_jacobian
    ijk: NDArray[np.float32]  # voxel ijk coordinates
    xyz: NDArray[np.float32]  # voxel xyz coordinates
    wavelengths: NDArray[np.int_]  # wavelengths
    resolution: Tuple[float, float, float]
    extinction_coefficients: Dict[int, ExtinctionCoefficient]  # HbO and HbR extinction coefficients


_assets: Assets | None = None

REG_DEFAULT = 0.1
RESHAPING_ORDER = "F"


def _load_assets(
    *,
    mega_jacobian_path: pathlib.Path | str,
    channel_mapping_path: pathlib.Path | str,
    coefficients_path: pathlib.Path | str,
    voxel_info_dir: pathlib.Path | str,
) -> Assets:
    """Load large reconstruction assets on demand.

    Parameters
    ----------
    mega_jacobian_path : pathlib.Path | str, optional
        Path to the mega Jacobian (.npy/.npz). Defaults to the module constant.
    channel_mapping_path : pathlib.Path | str, optional
        Path to the channel mapping JSON. Defaults to the module constant.
    voxel_info_dir : pathlib.Path | str, optional
        Directory containing voxel info files (mua, musp, idxs_significant_voxels, ijk, xyz, wavelengths).
        Defaults to the module constant.

    returns
    -------
    Assets
        Loaded reconstruction assets.
    """
    logger.info("Initializing reconstruction assets from disk.")

    mega_jacobian_path = pathlib.Path(mega_jacobian_path)
    channel_mapping_path = pathlib.Path(channel_mapping_path)

    # note no memmap because we'll offload to GPU, but this is slow
    _mega_jacobian = np.load(mega_jacobian_path).astype(np.float32, copy=False)

    with channel_mapping_path.open() as f:
        _channel_mapping = json.load(f)
    # load extinction coefficients
    _extinction_coefficients = ExtinctionCoefficient.from_csv(pathlib.Path(coefficients_path))
    # load mua, musp, idxs_significant_voxels, ijk, xyz, wavelengths
    mua_path = pathlib.Path(voxel_info_dir) / "mua.npy"
    musp_path = pathlib.Path(voxel_info_dir) / "musp.npy"
    idxs_significant_voxels_path = pathlib.Path(voxel_info_dir) / "idxs_significant_voxels.npy"
    ijk_path = pathlib.Path(voxel_info_dir) / "ijk.npy"
    xyz_path = pathlib.Path(voxel_info_dir) / "xyz.npy"
    wavelengths_path = pathlib.Path(voxel_info_dir) / "wavelengths.npy"
    resolution_path = pathlib.Path(voxel_info_dir) / "resolution.npy"

    _mua = np.load(pathlib.Path(mua_path))
    _musp = np.load(pathlib.Path(musp_path))
    _idxs_significant_voxels = np.load(pathlib.Path(idxs_significant_voxels_path))
    _ijk = np.load(pathlib.Path(ijk_path))
    _xyz = np.load(pathlib.Path(xyz_path))
    _wavelengths = np.load(pathlib.Path(wavelengths_path))
    _resolution = tuple(np.load(pathlib.Path(resolution_path)).tolist())

    logger.info("Reconstruction assets initialization complete.")

    return Assets(
        mega_jacobian=cast(NDArray[np.float32], _mega_jacobian),
        channel_mapping=cast(ChannelHeadsetMapping, _channel_mapping),
        mua=cast(NDArray[np.float32], _mua),
        musp=cast(NDArray[np.float32], _musp),
        extinction_coefficients=_extinction_coefficients,
        idxs_significant_voxels=cast(NDArray[np.int_], _idxs_significant_voxels),
        ijk=cast(NDArray[np.float32], _ijk),
        xyz=cast(NDArray[np.float32], _xyz),
        wavelengths=cast(NDArray[np.int_], _wavelengths),
        resolution=_resolution,
    )


def get_assets(
    jacobian_path: pathlib.Path | str,
    channel_mapping_path: pathlib.Path | str,
    voxel_info_dir: pathlib.Path | str,
    coefficients_path: pathlib.Path | str,
) -> Assets:
    global _assets
    if _assets is None:
        # load
        _assets = _load_assets(
            mega_jacobian_path=jacobian_path,
            channel_mapping_path=channel_mapping_path,
            voxel_info_dir=voxel_info_dir,
            coefficients_path=coefficients_path,
        )

    return _assets
