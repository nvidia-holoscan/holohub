"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0

BCI Visualization Application - streams synthetic voxel data and renders as 3D volume.
"""

import os
import argparse
from pathlib import Path


from holoscan.core import Application
from holoscan.operators import HolovizOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler, MultiThreadScheduler
# Import local operators
from operators.voxel_stream_to_volume import VoxelStreamToVolumeOp

from holohub.color_buffer_passthrough import ColorBufferPassthroughOp
from holohub.volume_renderer import VolumeRendererOp

class BciVisualizationApp(Application):
    """BCI Visualization Application with ClaraViz."""

    def __init__(self, 
        argv=None,
        *args,
        render_config_file,
        density_min,
        density_max,
        label_path=None,
        roi_labels=None,
        mask_path=None,
        **kwargs,
    ):
        self._rendering_config = render_config_file
        self._density_min = density_min
        self._density_max = density_max
        self._label_path = label_path
        self._roi_labels = roi_labels
        self._mask_path = mask_path

        super().__init__(argv, *args, **kwargs)

    def compose(self):
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

        # Selection
        selected_channel = 0  # 0: HbO, 1: HbR

        # Operators
        voxel_to_volume_args = {
            "selected_channel": selected_channel,
        }
        if self._label_path:
            voxel_to_volume_args["label_path"] = self._label_path
        if self._roi_labels:
            voxel_to_volume_args["roi_labels"] = self._roi_labels
        if self._mask_path:
            voxel_to_volume_args["mask_nifti_path"] = self._mask_path
            
        voxel_to_volume = VoxelStreamToVolumeOp(
            self,
            name="voxel_to_volume",
            pool=volume_allocator,
            **voxel_to_volume_args,
        )

        volume_renderer_args = {}
        if self._density_min:
            volume_renderer_args["density_min"] = self._density_min
        if self._density_max:
            volume_renderer_args["density_max"] = self._density_max

        volume_renderer = VolumeRendererOp(
            self,
            name="volume_renderer",
            config_file=self._rendering_config,
            allocator=volume_allocator,
            alloc_width=1024,
            alloc_height=768,
            cuda_stream_pool=cuda_stream_pool,
            **volume_renderer_args,
        )

        holoviz = HolovizOp(
            self,
            name="holoviz",
            window_title="BCI Visualization with ClaraViz",
            enable_camera_pose_output=True,
            cuda_stream_pool=cuda_stream_pool,
        )

        color_buffer_passthrough = ColorBufferPassthroughOp(
            self,
            name="color_buffer_passthrough",
        )

        # Connect operators

        kernel_data = Path("/workspace/holohub/data/kernel")
        from streams.snirf import SNIRFStream
        stream = SNIRFStream(kernel_data / "data.snirf")
        from reconstruction import ReconstructionApplication
        reconstruction_application = ReconstructionApplication(
            stream=stream,
            jacobian_path=kernel_data / "flow_mega_jacobian.npy",
            channel_mapping_path=kernel_data / "flow_channel_map.json",
            voxel_info_dir=kernel_data / "voxel_info",
            coefficients_path=kernel_data / "extinction_coefficients_mua.csv",
            use_gpu=True,
        )
        reconstruction_application.compose(self, voxel_to_volume)

        # voxel_to_volume → volume_renderer
        self.add_flow(voxel_to_volume, volume_renderer, {
            ("volume", "density_volume"),
            ("spacing", "density_spacing"),
            ("permute_axis", "density_permute_axis"),
            ("flip_axes", "density_flip_axes"),
        })
        # Add mask connections to VolumeRendererOp
        self.add_flow(voxel_to_volume, volume_renderer, {
            ("mask_volume", "mask_volume"),
            ("mask_spacing", "mask_spacing"),
            ("mask_permute_axis", "mask_permute_axis"),
            ("mask_flip_axes", "mask_flip_axes"),
        })

        # volume_renderer ↔ holoviz
        self.add_flow(volume_renderer, color_buffer_passthrough,
                      {("color_buffer_out", "color_buffer_in")})
        self.add_flow(color_buffer_passthrough, holoviz, {("color_buffer_out", "receivers")})
        self.add_flow(holoviz, volume_renderer, {("camera_pose_output", "camera_pose")})


def main():
    
    
    parser = argparse.ArgumentParser(description="BCI Visualization Application", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        dest="config",
        help="Name of the renderer JSON configuration file to load",
    )

    parser.add_argument(
        "-i",
        "--density_min",
        action="store",
        type=int,
        dest="density_min",
        help="Set the minimum of the density element values. If not set this is calculated from the"
        "volume data. In practice CT volumes have a minimum value of -1024 which corresponds to"
        "the lower value of the Hounsfield scale range usually used.",
    )
    parser.add_argument(
        "-a",
        "--density_max",
        action="store",
        type=int,
        dest="density_max",
        help="Set the maximum of the density element values. If not set this is calculated from the"
        "volume data. In practice CT volumes have a maximum value of 3071 which corresponds to"
        "the upper value of the Hounsfield scale range usually used.",
    )
    
    parser.add_argument(
        "-l",
        "--label_path",
        action="store",
        type=str,
        dest="label_path",
        help="Path to the NPZ file containing brain anatomy labels. If not provided, uses default path.",
    )
    
    parser.add_argument(
        "-r",
        "--roi_labels",
        action="store",
        type=str,
        dest="roi_labels",
        help="Comma-separated list of label values to use as ROI (e.g., '3,4'). Default is '3,4'.",
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
    
    # Parse roi_labels from comma-separated string to list of integers
    roi_labels = None
    if args.roi_labels:
        try:
            roi_labels = [int(label.strip()) for label in args.roi_labels.split(',')]
        except ValueError:
            print(f"Warning: Invalid roi_labels format '{args.roi_labels}'. Expected comma-separated integers.")
            roi_labels = None

    app = BciVisualizationApp(
        render_config_file=args.config,
        density_min=args.density_min,
        density_max=args.density_max,
        label_path=args.label_path,
        roi_labels=roi_labels,
        mask_path=args.mask_path,
    )

    app.scheduler(EventBasedScheduler(app, worker_thread_number=5, stop_on_deadlock=True))

    app.run()
    
    print("BCI Visualization Application has finished running.")


if __name__ == "__main__":
    main()

