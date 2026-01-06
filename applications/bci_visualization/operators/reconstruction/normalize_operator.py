"""
SPDX-FileCopyrightText: Copyright (c) 2026 Kernel.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from holoscan.core import (
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from .types import BuildRHSOutput, NormalizedSolveBatch, WavelengthSystem

logger = logging.getLogger(__name__)


HARD_CODED_NORMALIZERS = [  # for each feature type (moment)
    np.array([1, 5e2, 5e5]),
    np.array([0.5, 2.5e2, 2.5e5]),
]

class NormalizeOperator(Operator):
    """Apply Jacobian/RHS normalization before solver execution."""

    def __init__(
        self,
        *,
        fragment: Any | None = None,
        use_gpu: bool = False,
        use_hard_coded_normalizers: bool = True,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)
        self._jacobian_cache: NDArray[np.float32] | None = None
        self._use_gpu = use_gpu
        self._max_rhs: NDArray[np.float32] | None = None

        self._use_hard_coded_normalizers = use_hard_coded_normalizers
        self._hard_coded_row_normalizers_cache: NDArray[np.float32] | None = None
        self._hard_coded_normalized_jacobian_cache: NDArray[np.float32] | None = None

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("batch")
        spec.output("normalized")

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        batch = op_input.receive("batch")
        if not isinstance(batch, BuildRHSOutput):
            raise TypeError(f"NormalizeOperator expected BuildRHSOutput, got {type(batch)}")

        cuda_stream = op_input.receive_cuda_stream("batch")

        with cp.cuda.ExternalStream(cuda_stream):
            result = self._normalize_batch(batch)
            if result is None:
                logger.info("Skipping normalization for frame because max_rhs is all zeros")
                return

        systems, num_absorbers = result

        op_output.emit(
            NormalizedSolveBatch(
                systems=tuple(systems),
                idxs_significant_voxels=batch.idxs_significant_voxels,
                num_full_voxels=batch.num_full_voxels,
                num_absorbers=num_absorbers,
                wavelengths=batch.wavelengths,
                voxel_metadata=batch.voxel_metadata,
            ),
            "normalized",
        )

    def _get_hard_coded_row_normalizers(
        self,
        num_rows: int,
        num_features: int,
        num_wavelengths: int,
    ) -> np.ndarray:
        if self._hard_coded_row_normalizers_cache is not None:
            return self._hard_coded_row_normalizers_cache

        row_normalizers = cp.full((num_rows, num_wavelengths), cp.nan)
        for wavelength_idx in range(num_wavelengths):
            for idx_feature in range(num_features):
                row_normalizers[idx_feature::num_features, wavelength_idx] = HARD_CODED_NORMALIZERS[
                    wavelength_idx
                ][idx_feature]

        assert not cp.any(cp.isnan(row_normalizers))
        self._hard_coded_row_normalizers_cache = row_normalizers
        return row_normalizers

    def _normalize_batch(self, batch: BuildRHSOutput) -> Tuple[List[WavelengthSystem], int] | None:
        num_cols = batch.data_jacobians.shape[-1]
        num_significant = int(batch.idxs_significant_voxels.size)
        num_absorbers, remainder = divmod(num_cols, num_significant)
        assert not remainder

        # normalize rows
        rhs = cp.asarray(batch.data_rhs, dtype=cp.float32)
        num_wavelengths = batch.data_rhs.shape[-1]
        row_normalizers = self._get_hard_coded_row_normalizers(
            batch.data_jacobians.shape[0], batch.num_features, num_wavelengths
        )

        jacobian_template = self._get_template_jacobians(batch)
        if (
            self._use_hard_coded_normalizers
            and self._hard_coded_normalized_jacobian_cache is not None
        ):
            jacobians = self._hard_coded_normalized_jacobian_cache
        else:
            jacobians = jacobian_template.copy()
            jacobians /= row_normalizers[:, :, None]

        rhs /= row_normalizers

        if self._use_hard_coded_normalizers and self._hard_coded_normalized_jacobian_cache is None:
            self._hard_coded_normalized_jacobian_cache = jacobians

        systems: List[WavelengthSystem] = []
        for idx_wavelength in range(num_wavelengths):
            background_payload = cp.asarray(
                batch.model_optical_properties[:, idx_wavelength],
                dtype=cp.float32,
            )
            systems.append(
                WavelengthSystem(
                    jacobian=jacobians[:, idx_wavelength, :],
                    rhs=rhs[:, idx_wavelength],
                    background=background_payload,
                )
            )

        return systems, num_absorbers

    def _get_template_jacobians(
        self,
        batch: BuildRHSOutput,
    ) -> NDArray[np.float32]:
        """
        Retrieve or build the normalized Jacobian template for the given batch.
        """
        if self._jacobian_cache is not None:
            return self._jacobian_cache

        template = cp.asarray(batch.data_jacobians, dtype=cp.float32).copy()
        background = cp.asarray(batch.model_optical_properties, dtype=cp.float32)

        background_T = cp.swapaxes(background, 0, 1)
        template *= background_T[None, :, :]

        self._jacobian_cache = template
        return template
