"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VoxelMetadata:
    ijk: NDArray[np.float32]
    xyz: NDArray[np.float32] | None
    resolution: Tuple[float, float, float]


@dataclass(frozen=True)
class BuildRHSOutput:
    data_jacobians: NDArray[np.float32]  # 3d array (feature-channels, wavelengths, voxels)
    data_rhs: NDArray[np.float32]  # 2d array (feature-channels, wavelengths)
    model_optical_properties: NDArray[np.float32]
    idxs_significant_voxels: NDArray[np.int_]
    num_full_voxels: int
    num_features: int
    wavelengths: Tuple[int, ...]
    voxel_metadata: VoxelMetadata


@dataclass(frozen=True)
class WavelengthSystem:
    jacobian: NDArray[np.float32]
    rhs: NDArray[np.float32]
    background: NDArray[np.float32]


@dataclass(frozen=True)
class NormalizedSolveBatch:
    systems: Tuple[WavelengthSystem, ...]
    idxs_significant_voxels: NDArray[np.int_]
    num_full_voxels: int
    num_absorbers: int
    wavelengths: Tuple[int, ...]
    voxel_metadata: VoxelMetadata


@dataclass(frozen=True)
class SolverResult:
    data_mua: NDArray[np.float32]
    data_musp: NDArray[np.float32]
    idxs_significant_voxels: NDArray[np.int_]
    num_full_voxels: int
    wavelengths: Tuple[int, ...]
    voxel_metadata: VoxelMetadata
