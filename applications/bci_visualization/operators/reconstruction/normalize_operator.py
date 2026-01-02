from __future__ import annotations

import logging
from types import ModuleType
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
        xp: ModuleType,
        num_rows: int,
        num_features: int,
        num_wavelengths: int,
    ) -> np.ndarray:
        if self._hard_coded_row_normalizers_cache is not None:
            return self._hard_coded_row_normalizers_cache

        row_normalizers = xp.full((num_rows, num_wavelengths), xp.nan)
        for wavelength_idx in range(num_wavelengths):
            for idx_feature in range(num_features):
                row_normalizers[idx_feature::num_features, wavelength_idx] = HARD_CODED_NORMALIZERS[
                    wavelength_idx
                ][idx_feature]

        assert not xp.any(xp.isnan(row_normalizers))
        self._hard_coded_row_normalizers_cache = row_normalizers
        return row_normalizers

    def _compute_row_normalizers(
        self,
        xp: ModuleType,
        rhs: NDArray[np.float32],
        num_rows: int,
        num_features: int,
        num_wavelengths: int,
    ) -> NDArray[np.float32] | None:
        if self._use_hard_coded_normalizers:
            return self._get_hard_coded_row_normalizers(
                xp,
                num_rows,
                num_features,
                num_wavelengths,
            )

        # We store ONE max value per wavelength/feature type to preserve relative contrast
        if self._max_rhs is None:
            self._max_rhs = xp.zeros((num_features, num_wavelengths), dtype=xp.float32)
            return None  # early return because normalizer of all 0 is invalid for solving

        # Iterate through features to pool the max across all relevant rows
        for wavelength_idx in range(num_wavelengths):
            for idx_feature in range(num_features):
                # Extract all rows and wavelengths for this feature
                # Shape: (subset_rows, wavelengths)
                feature_data = xp.abs(rhs[idx_feature::num_features, wavelength_idx])

                # Find the single highest value in this batch for this entire feature type
                batch_feature_max = xp.max(feature_data)

                # Update the historical running max
                self._max_rhs[idx_feature, wavelength_idx] = max(
                    self._max_rhs[idx_feature, wavelength_idx], batch_feature_max
                )

        # build normalizer with rolling max, per feature and wavelength
        row_normalizers = xp.full((num_rows, num_wavelengths), xp.nan)
        for wavelength_idx in range(num_wavelengths):
            for idx_feature in range(num_features):
                row_normalizers[idx_feature::num_features, wavelength_idx] = self._max_rhs[
                    idx_feature, wavelength_idx
                ]

        return row_normalizers

    def _normalize_batch(self, batch: BuildRHSOutput) -> Tuple[List[WavelengthSystem], int] | None:
        num_cols = batch.data_jacobians.shape[-1]
        num_significant = int(batch.idxs_significant_voxels.size)
        num_absorbers, remainder = divmod(num_cols, num_significant)
        assert not remainder

        # GPU-only: always use CuPy, and rely on upstream CUDA stream propagation for correctness.
        xp = cp

        # normalize rows
        rhs = xp.asarray(batch.data_rhs, dtype=xp.float32)
        num_wavelengths = batch.data_rhs.shape[-1]
        row_normalizers = self._compute_row_normalizers(
            xp, rhs, batch.data_jacobians.shape[0], batch.num_features, num_wavelengths
        )
        if row_normalizers is None:
            return  # early exit if max_rhs is all zeros

        jacobian_template = self._get_template_jacobians(xp, batch)
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
            background_payload = xp.asarray(
                batch.model_optical_properties[:, idx_wavelength],
                dtype=xp.float32,
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
        xp: ModuleType,
        batch: BuildRHSOutput,
    ) -> NDArray[np.float32]:
        """
        Retrieve or build the normalized Jacobian template for the given batch.
        """
        if self._jacobian_cache is not None:
            return self._jacobian_cache

        template = xp.asarray(batch.data_jacobians, dtype=xp.float32).copy()
        background = xp.asarray(batch.model_optical_properties, dtype=xp.float32)

        background_T = xp.swapaxes(background, 0, 1)
        template *= background_T[None, :, :]

        self._jacobian_cache = template
        return template
