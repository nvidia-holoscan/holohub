from __future__ import annotations

from pathlib import Path

from processing.reconstruction import (
    REG_DEFAULT,
    get_assets,
)
from streams.base_nirs import BaseNirsStream
from holoscan.core import Application, Operator
from operators.reconstruction import (
    BuildRHSOperator,
    ConvertToVoxelsOperator,
    NormalizeOperator,
    RegularizedSolverOperator,
)
from operators.stream import StreamOperator


class ReconstructionApplication(Application):
    """Holoscan graph wiring KTensor/KSDK streams into the Reconstruction pipeline."""

    def __init__(
        self,
        *,
        stream: BaseNirsStream,
        jacobian_path: Path | str,
        channel_mapping_path: Path | str,
        voxel_info_dir: Path,
        coefficients_path: Path | str,
        reg: float = REG_DEFAULT,
        tol: float = 1e-4,  # Tolerance for the regularized solver
        use_gpu: bool = False,
    ):
        super().__init__()
        self._stream = stream
        self._reg = reg
        self._jacobian_path = Path(jacobian_path)
        self._channel_mapping_path = Path(channel_mapping_path)
        self._coefficients_path = Path(coefficients_path)

        self._voxel_info_dir = Path(voxel_info_dir)
        self._tol = tol
        self._use_gpu = use_gpu

    def compose(self, application: Application, voxel_to_volume: Operator):
        fragment = application
        emit_voxel_frames = True

        pipeline_assets = get_assets(
            jacobian_path=self._jacobian_path,
            channel_mapping_path=self._channel_mapping_path,
            voxel_info_dir=self._voxel_info_dir,
            coefficients_path=self._coefficients_path,
        )

        stream_operator = StreamOperator(stream=self._stream, fragment=fragment)
        build_rhs_operator = BuildRHSOperator(
            assets=pipeline_assets,
            fragment=fragment,
        )
        normalize_operator = NormalizeOperator(
            fragment=fragment,
            use_gpu=self._use_gpu,
        )
        regularized_solver_operator = RegularizedSolverOperator(
            reg=self._reg,
            use_gpu=self._use_gpu,
            fragment=fragment,
        )
        convert_to_voxels_operator = ConvertToVoxelsOperator(
            fragment=fragment,
            coefficients=pipeline_assets.extinction_coefficients,
            ijk=pipeline_assets.ijk,
            xyz=pipeline_assets.xyz,
            use_gpu=self._use_gpu,
        )

        application.add_flow(stream_operator, build_rhs_operator, {
           ("samples", "moments"),
        })
        application.add_flow(build_rhs_operator, normalize_operator, {
            ("batch", "batch"),
        })
        application.add_flow(normalize_operator, regularized_solver_operator, {
            ("normalized", "batch"),
        })
        application.add_flow(regularized_solver_operator, convert_to_voxels_operator, {
            ("result", "result"),
        })
        application.add_flow(convert_to_voxels_operator, voxel_to_volume, {
            ("affine_4x4", "affine_4x4"),
            ("hb_voxel_data", "hb_voxel_data"),
        })
