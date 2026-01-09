"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, cast

import cupy as cp
import numpy as np
from holoscan.core import ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext
from numpy.typing import NDArray
from utils.reconstruction.hbo import ExtinctionCoefficient, HbO

from .types import SolverResult, VoxelMetadata

logger = logging.getLogger(__name__)


def _convert_to_full_voxels(
    array: NDArray[np.float32],
    num_voxels: int,
    idxs_significant_voxels: NDArray[np.int_],
) -> NDArray[np.float32]:
    """Convert an array from significant voxels to full voxel arrays.

    Parameters
    ----------
    array : NDArray[np.float32]
        Array to convert from significant voxels to full voxels.
    num_voxels : int
        Number of total voxels in the mesh.
    idxs_significant_voxels : NDArray[np.int_]
        Indices of significant voxels.

    Returns
    -------
    array_full_voxels : NDArray[np.float32]
        The input array converted to full voxel space.
    """
    array_full_voxels = cp.zeros((num_voxels, array.shape[1]))
    array_full_voxels[idxs_significant_voxels, :] = array

    return array_full_voxels


def _compute_affine(xyz: NDArray[np.float32], ijk: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes affine based on coordinates.

    Parameters
    ----------
    xyz: NDArray[np.float32]
        x, y, z coordinates
    ijk: NDArray[np.float32]
        Voxel indexes
    Returns
    -------
    affine: np.ndarray
        Affine matrix defining the given space.
    """
    rng = np.random.default_rng(0)

    n = 4
    ctr = 0
    out: NDArray[np.float32] = np.array([])  # bind outside loop
    B: NDArray[np.float32] = np.array([])  # bind outside loop
    while ctr < 100:
        ctr += 1
        inds = rng.choice(np.arange(len(ijk)), size=n, replace=False)
        ins = ijk[np.array(inds), :]  # <- points
        out = xyz[np.array(inds), :]  # <- mapped to
        B = np.vstack([np.transpose(ins), np.ones(n, dtype=np.float32)])
        if np.linalg.det(B) == 0:
            continue
    if np.linalg.det(B) == 0:
        raise RuntimeError("Cannot compute affine, algorithm failed after 100 attempts")
    D = 1.0 / np.linalg.det(B)

    def entry(r, d):
        return np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))

    M = [[(-1) ** i * D * entry(R, i) for i in range(n)] for R in np.transpose(out)]

    affine = np.concatenate((M, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)
    assert affine.shape == (4, 4)
    return affine


class ConvertToVoxelsOperator(Operator):
    """Expand trimmed solver outputs back to the full voxel grid."""

    def __init__(
        self,
        *,
        coefficients: Dict[int, ExtinctionCoefficient],
        ijk: NDArray[np.float32],
        xyz: NDArray[np.float32],
        fragment: Any | None = None,
    ) -> None:
        self._hbo = HbO(coefficients)
        self._affine = np.round(_compute_affine(xyz, ijk), 6)
        self._affine_sent: bool = False
        self._cached_affine: NDArray[np.float32] | None = None
        self._cum_hbo: NDArray[np.float32] | None = None
        self._cum_hbr: NDArray[np.float32] | None = None

        super().__init__(fragment, name=self.__class__.__name__)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("result")
        spec.output("affine_4x4")
        spec.output("hb_voxel_data")

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        result: SolverResult = op_input.receive("result")
        cuda_stream = op_input.receive_cuda_stream("result")

        with cp.cuda.ExternalStream(cuda_stream):
            data_mua_full = _convert_to_full_voxels(
                result.data_mua,
                result.num_full_voxels,
                result.idxs_significant_voxels,
            )

            data_hbo, data_hbr = self._hbo.convert_mua_to_hb(
                data_mua_full,
                result.wavelengths,
                result.idxs_significant_voxels,
            )

            self._cum_hbo = data_hbo if self._cum_hbo is None else self._cum_hbo + data_hbo
            self._cum_hbr = data_hbr if self._cum_hbr is None else self._cum_hbr + data_hbr

            layout = self._compute_voxel_layout(result.voxel_metadata)
            hb_volume = self._voxelize_hbo(self._cum_hbo, layout)

        self._emit_affine_once(op_output)
        op_output.emit(hb_volume, "hb_voxel_data")

    def _emit_affine_once(self, op_output: OutputContext) -> None:
        if self._affine_sent:
            return

        op_output.emit(self._affine, "affine_4x4")
        self._affine_sent = True

    def _voxelize_hbo(
        self,
        data_hbo: NDArray[np.float32],
        layout: Tuple[NDArray[np.int_], Tuple[int, int, int], NDArray[np.int_]],
    ) -> NDArray[np.float32]:
        scatter_coords, normalized_shape, _ijk_int = layout
        scatter_coords = scatter_coords.astype(np.int32, copy=False)  # for indexing

        num_voxels = data_hbo.shape[0]
        assert num_voxels == scatter_coords.shape[0]

        # scatter ijk to full voxel grid
        volume_small: NDArray[np.float32] = cp.zeros(normalized_shape, dtype=data_hbo.dtype)
        x_idx, y_idx, z_idx = scatter_coords.T
        volume_small[x_idx, y_idx, z_idx] = data_hbo

        return volume_small

    def _compute_voxel_layout(
        self,
        metadata: VoxelMetadata,
    ) -> Tuple[NDArray[np.int_], Tuple[int, int, int], NDArray[np.int_]]:
        """
        Compute normalized voxel coordinates and grid shape from metadata.
        """
        ijk = cp.asarray(metadata.ijk)
        assert ijk.ndim == 2 and ijk.shape[1] == 3

        ijk_int = cp.rint(ijk)
        min_idx = ijk_int.min(axis=0)
        normalized = ijk_int - min_idx
        shape = tuple(int(axis_max) + 1 for axis_max in normalized.max(axis=0))
        assert all(dim > 0 for dim in shape)
        return normalized, cast(Tuple[int, int, int], shape), ijk_int
