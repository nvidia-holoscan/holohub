from __future__ import annotations

import logging
from typing import Any, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from processing.reconstruction import (
    Assets,
    get_channel_mask,
)
from holoscan.core import (
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from ..stream import SampleOutput
from .types import BuildRHSOutput, VoxelMetadata

logger = logging.getLogger(__name__)


class BuildRHSOperator(Operator):
    """Convert realtime moments tensors into trimmed RHS/Jacobian batches."""

    def __init__(
        self,
        *,
        assets: Assets,
        fragment: Any | None = None,
    ) -> None:
        super().__init__(fragment, name=self.__class__.__name__)

        self._model_optical_properties = np.concatenate((assets.mua, assets.musp)).astype(
            np.float32, copy=False
        )
        self._mega_jacobians = assets.mega_jacobian
        self._channel_mapping = assets.channel_mapping
        self._idxs_significant_voxels = assets.idxs_significant_voxels
        self._voxel_metadata = VoxelMetadata(
            ijk=assets.ijk, xyz=assets.xyz, resolution=assets.resolution
        )
        self._wavelengths = assets.wavelengths
        self._jacobian_cache: NDArray[np.float32] | None = None
        self._last_frame: NDArray[np.float32] | None = None  # previous frame

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("moments")
        spec.output("batch")

    def _apply_baseline(self, realtime_moments: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """
        simple diff against previous frame
        """
        if self._last_frame is None:
            self._last_frame = np.array(realtime_moments, copy=True)
            return None

        # diff with last frame and update last frame
        diff = realtime_moments - self._last_frame
        self._last_frame = np.array(realtime_moments, copy=True)
        return diff

    def _zero_out_invalids(self, data_rhs: NDArray[np.float32]) -> None:
        invalid_samples = ~np.isfinite(data_rhs)
        if not np.any(invalid_samples):
            return

        np.nan_to_num(data_rhs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ) -> None:
        del context

        payload: SampleOutput = op_input.receive("moments")
        realtime_moments = payload.data

        # take log of moment 0 to convert to optical density
        # shape is (moments, channels, wavelengths)
        np.log(realtime_moments[0], out=realtime_moments[0])

        realtime_moments = self._apply_baseline(realtime_moments)
        if realtime_moments is None:
            logger.info("Skipping RHS build for first frame (baseline capture)")
            return

        flowaxis_optodes: List[Tuple[int, int, int, int]] = [
            (
                payload.channels.source_module[channel_idx],
                payload.channels.source_number[channel_idx],
                payload.channels.detector_module[channel_idx],
                payload.channels.detector_number[channel_idx],
            )
            for channel_idx in range(len(payload.channels))
        ]

        # Validate that jacobian features dimension matches realtime moments
        # 5D jacobian shape: (channels, features, wavelengths, voxels, simulation_types)
        num_features = realtime_moments.shape[0]
        assert self._mega_jacobians.shape[1] == num_features

        if self._jacobian_cache is None:
            channel_mask = get_channel_mask(
                optode_order=flowaxis_optodes,
                headset_mapping=self._channel_mapping,
                mask_size=self._mega_jacobians.shape[0],
            )

            jacobians = self._mega_jacobians[channel_mask, :, :, :, :]  # 5d

            # swap axes so it's features, channels first
            jacobians = jacobians.transpose(1, 0, 2, 3, 4)
            # reshape to 3d and use F ordering
            # data_jacobians is ( num_channels, num_features, num_wavelengths, num_voxels, num_simulation_types)
            # we want to reshape to (features x channels, wavelengths, significant voxels x num_simulation_types)
            jacobians = np.reshape(
                jacobians,
                (
                    jacobians.shape[0] * jacobians.shape[1],
                    jacobians.shape[2],
                    jacobians.shape[3] * jacobians.shape[4],
                ),
                order="F",
            )
            self._jacobian_cache = jacobians

        data_jacobians = self._jacobian_cache
        # swap from (moments, channels, wavelengths) to (channels, moments, wavelengths)
        # then reshape to (channels x moments, wavelengths)
        data_rhs = realtime_moments.transpose(1, 0, 2).reshape(-1, realtime_moments.shape[2])

        self._zero_out_invalids(data_rhs)

        op_output.emit(
            BuildRHSOutput(
                data_jacobians=data_jacobians,
                data_rhs=data_rhs,
                model_optical_properties=cast(np.ndarray, self._model_optical_properties),
                idxs_significant_voxels=self._idxs_significant_voxels,
                num_full_voxels=self._voxel_metadata.ijk.shape[0],
                num_features=num_features,
                wavelengths=tuple(self._wavelengths.tolist()),
                voxel_metadata=self._voxel_metadata,
            ),
            "batch",
        )
