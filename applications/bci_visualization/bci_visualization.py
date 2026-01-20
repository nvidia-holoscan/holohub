"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

BCI Visualization Application - streams synthetic voxel data and renders as 3D volume.
"""

import argparse
import os
from pathlib import Path

from holoscan.core import Application, ConditionType
from holoscan.operators import HolovizOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler
from operators.reconstruction import (
    BuildRHSOperator,
    ConvertToVoxelsOperator,
    NormalizeOperator,
    RegularizedSolverOperator,
)
from operators.stream import StreamOperator
from operators.voxel_stream_to_volume import VoxelStreamToVolumeOp
from streams.base_nirs import BaseNirsStream
from streams.kernel_sdk import KernelSDKStream
from streams.snirf import SNIRFStream
from utils.reconstruction.assets import get_assets

from holohub.color_buffer_passthrough import ColorBufferPassthroughOp
from holohub.volume_renderer import VolumeRendererOp


class BciVisualizationApp(Application):
    """Kernel Flow BCI Real-Time Reconstruction and Visualization

    This application integrates the full pipeline from NIRS data streaming through
    reconstruction to 3D volume rendering and visualization.
    """

    def __init__(
        self,
        argv=None,
        *args,
        render_config_file,
        stream: BaseNirsStream,
        jacobian_path: Path | str,
        channel_mapping_path: Path | str,
        voxel_info_dir: Path | str,
        coefficients_path: Path | str,
        mask_path=None,
        reg: float = RegularizedSolverOperator.REG_DEFAULT,
        **kwargs,
    ):
        self._rendering_config = render_config_file
        self._mask_path = mask_path

        # Reconstruction pipeline parameters
        self._stream = stream
        self._reg = reg
        self._jacobian_path = Path(jacobian_path)
        self._channel_mapping_path = Path(channel_mapping_path)
        self._coefficients_path = Path(coefficients_path)
        self._voxel_info_dir = Path(voxel_info_dir)

        super().__init__(argv, *args, **kwargs)

    def compose(self):
        # Resources
        volume_allocator = UnboundedAllocator(self, name="allocator")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Get reconstruction pipeline assets
        pipeline_assets = get_assets(
            jacobian_path=self._jacobian_path,
            channel_mapping_path=self._channel_mapping_path,
            voxel_info_dir=self._voxel_info_dir,
            coefficients_path=self._coefficients_path,
        )

        # ========== Reconstruction Pipeline Operators ==========
        stream_operator = StreamOperator(stream=self._stream, fragment=self)

        build_rhs_operator = BuildRHSOperator(
            assets=pipeline_assets,
            fragment=self,
        )

        normalize_operator = NormalizeOperator(
            fragment=self,
        )

        regularized_solver_operator = RegularizedSolverOperator(
            reg=self._reg,
            fragment=self,
        )

        convert_to_voxels_operator = ConvertToVoxelsOperator(
            fragment=self,
            coefficients=pipeline_assets.extinction_coefficients,
            ijk=pipeline_assets.ijk,
            xyz=pipeline_assets.xyz,
        )

        # ========== Visualization Pipeline Operators ==========
        # Get volume_renderer kwargs from YAML config to extract density range
        volume_renderer_kwargs = self.kwargs("volume_renderer")
        density_min = volume_renderer_kwargs.get("density_min", -100.0)
        density_max = volume_renderer_kwargs.get("density_max", 100.0)

        voxel_to_volume = VoxelStreamToVolumeOp(
            self,
            name="voxel_to_volume",
            pool=volume_allocator,
            mask_nifti_path=self._mask_path,
            density_min=density_min,
            density_max=density_max,
            **self.kwargs("voxel_stream_to_volume"),
        )

        volume_renderer = VolumeRendererOp(
            self,
            name="volume_renderer",
            config_file=self._rendering_config,
            allocator=volume_allocator,
            cuda_stream_pool=cuda_stream_pool,
            **volume_renderer_kwargs,
        )

        # IMPORTANT changes to avoid deadlocks of volume_renderer and holoviz
        # when running in multi-threading mode
        # 1. Set the output port condition to NONE to remove backpressure
        volume_renderer.spec.outputs["color_buffer_out"].condition(ConditionType.NONE)
        # 2. Use a passthrough operator to configure queue policy as POP to use latest frame
        color_buffer_passthrough = ColorBufferPassthroughOp(
            self,
            name="color_buffer_passthrough",
        )

        holoviz = HolovizOp(
            self,
            name="holoviz",
            window_title="Kernel Flow BCI Real-Time Reconstruction and Visualization",
            enable_camera_pose_output=True,
            cuda_stream_pool=cuda_stream_pool,
        )

        # ========== Connect Reconstruction Pipeline ==========
        self.add_flow(
            stream_operator,
            build_rhs_operator,
            {
                ("samples", "moments"),
            },
        )
        self.add_flow(
            build_rhs_operator,
            normalize_operator,
            {
                ("batch", "batch"),
            },
        )
        self.add_flow(
            normalize_operator,
            regularized_solver_operator,
            {
                ("normalized", "batch"),
            },
        )
        self.add_flow(
            regularized_solver_operator,
            convert_to_voxels_operator,
            {
                ("result", "result"),
            },
        )
        self.add_flow(
            convert_to_voxels_operator,
            voxel_to_volume,
            {
                ("affine_4x4", "affine_4x4"),
                ("hb_voxel_data", "hb_voxel_data"),
            },
        )

        # ========== Connect Visualization Pipeline ==========
        # voxel_to_volume → volume_renderer
        self.add_flow(
            voxel_to_volume,
            volume_renderer,
            {
                ("volume", "density_volume"),
                ("spacing", "density_spacing"),
                ("permute_axis", "density_permute_axis"),
                ("flip_axes", "density_flip_axes"),
            },
        )
        # Add mask connections to VolumeRendererOp
        self.add_flow(
            voxel_to_volume,
            volume_renderer,
            {
                ("mask_volume", "mask_volume"),
                ("mask_spacing", "mask_spacing"),
                ("mask_permute_axis", "mask_permute_axis"),
                ("mask_flip_axes", "mask_flip_axes"),
            },
        )

        # volume_renderer ↔ holoviz
        self.add_flow(
            volume_renderer, color_buffer_passthrough, {("color_buffer_out", "color_buffer_in")}
        )
        self.add_flow(color_buffer_passthrough, holoviz, {("color_buffer_out", "receivers")})
        self.add_flow(holoviz, volume_renderer, {("camera_pose_output", "camera_pose")})


def main():
    parser = argparse.ArgumentParser(
        description="Kernel Flow BCI Real-Time Reconstruction and Visualization", add_help=False
    )

    parser.add_argument(
        "-c",
        "--renderer_config",
        action="store",
        dest="renderer_config",
        help="Path to the renderer JSON configuration file to load",
    )

    parser.add_argument(
        "-m",
        "--mask_path",
        action="store",
        type=str,
        dest="mask_path",
        help="Path to the mask NIfTI file containing 3D integer labels (I, J, K). Optional.",
    )

    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Help message"
    )

    args = parser.parse_args()

    # Setup data paths
    default_data_path = os.path.join(os.getcwd(), "data/bci_visualization")
    kernel_data = Path(os.environ.get("HOLOSCAN_INPUT_PATH", default_data_path))

    stream = None
    if os.environ.get("KERNEL_SDK") == "1":
        stream = KernelSDKStream()
    else:
        stream = SNIRFStream(kernel_data / "data.snirf")

    app = BciVisualizationApp(
        render_config_file=args.renderer_config,
        stream=stream,
        jacobian_path=kernel_data / "flow_mega_jacobian.npy",
        channel_mapping_path=kernel_data / "flow_channel_map.json",
        voxel_info_dir=kernel_data / "voxel_info",
        coefficients_path=kernel_data / "extinction_coefficients_mua.csv",
        mask_path=args.mask_path,
        reg=RegularizedSolverOperator.REG_DEFAULT,
    )

    # Load YAML configuration
    config_file = os.path.join(os.path.dirname(__file__), "bci_visualization.yaml")

    app.config(config_file)
    app.scheduler(EventBasedScheduler(app, worker_thread_number=5, stop_on_deadlock=True))

    app.run()

    print("App has finished running.")


if __name__ == "__main__":
    main()
