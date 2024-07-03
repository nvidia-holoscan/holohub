# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pathlib

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp
from holoscan.resources import UnboundedAllocator

from holohub.volume_loader import VolumeLoaderOp
from holohub.volume_renderer import VolumeRendererOp

logger = logging.getLogger("volume_rendering")


class VolumeRenderingApp(Application):
    def __init__(
        self, argv=None, *args, render_config_file, density_volume_file, mask_volume_file, **kwargs
    ):
        self._rendering_config = render_config_file
        self._density_volume_file = density_volume_file
        self._mask_volume_file = mask_volume_file

        super().__init__(argv, *args, **kwargs)

    def compose(self):

        volume_allocator = UnboundedAllocator(self, name="allocator")

        density_volume_loader = VolumeLoaderOp(
            self,
            CountCondition(self, count=1),
            name="density_volume_loader",
            file_name=self._density_volume_file,
            allocator=volume_allocator,
        )

        mask_volume_loader = VolumeLoaderOp(
            self,
            CountCondition(self, count=1),
            file_name=self._mask_volume_file,
            name="mask_volume_loader",
            allocator=volume_allocator,
        )

        volume_renderer = VolumeRendererOp(
            self,
            name="volume_renderer",
            config_file=self._rendering_config,
            allocator=volume_allocator,
            alloc_width=1024,
            alloc_height=768,
        )

        # Python is not supporting gxf::VideoBuffer, need to convert the video buffer received
        # from the volume renderer to a tensor.
        volume_renderer_format_converter = FormatConverterOp(
            self,
            name="volume_renderer_format_converter",
            pool=volume_allocator,
            in_dtype="rgba8888",
            out_dtype="rgba8888",
        )
        visualizer = HolovizOp(
            self,
            name="viz",
            window_title="Volume Rendering with ClaraViz",
            enable_camera_pose_output=True,
        )

        self.add_flow(
            density_volume_loader,
            volume_renderer,
            {
                ("volume", "density_volume"),
                ("spacing", "density_spacing"),
                ("permute_axis", "density_permute_axis"),
                ("flip_axes", "density_flip_axes"),
            },
        )

        self.add_flow(
            mask_volume_loader,
            volume_renderer,
            {
                ("volume", "mask_volume"),
                ("spacing", "mask_spacing"),
                ("permute_axis", "mask_permute_axis"),
                ("flip_axes", "mask_flip_axes"),
            },
        )

        self.add_flow(
            volume_renderer,
            volume_renderer_format_converter,
            {("color_buffer_out", "source_video")},
        )

        self.add_flow(volume_renderer_format_converter, visualizer, {("tensor", "receivers")})
        self.add_flow(visualizer, volume_renderer, {("camera_pose_output", "camera_matrix")})


def valid_existing_path(path: str) -> pathlib.Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given file/folder path exists.

    Args:
        path: string input path

    Returns:
        If path exists, return absolute path as a pathlib.Path object.

        If path doesn't exist, raises argparse.ArgumentTypeError.
    """
    path = os.path.expanduser(path)
    file_path = pathlib.Path(path).absolute()
    if file_path.exists():
        return file_path
    raise argparse.ArgumentTypeError(f"No such file/folder: '{file_path}'")


if __name__ == "__main__":
    render_config_file_default = pathlib.Path(
        "../../../data/volume_rendering/config.json"
    ).resolve()
    density_volume_file_default = pathlib.Path(
        "../../../data/volume_rendering/highResCT.mhd"
    ).resolve()
    mask_volume_file_default = pathlib.Path(
        "../../../data/volume_rendering/smoothmasks.seg.mhd"
    ).resolve()

    parser = argparse.ArgumentParser(description="Volume Rendering Application", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default=render_config_file_default,
        type=valid_existing_path,
        dest="config",
        help=f"Name of the renderer JSON configuration file to load (default {render_config_file_default})",
    )
    parser.add_argument(
        "-d",
        "--density",
        action="store",
        default=density_volume_file_default,
        type=valid_existing_path,
        dest="density",
        help=f"Name of density volume file to load (default {density_volume_file_default})",
    )
    parser.add_argument(
        "-m",
        "--mask",
        action="store",
        default=mask_volume_file_default,
        type=valid_existing_path,
        dest="mask",
        help=f"Name of mask volume file to load (default {mask_volume_file_default})",
    )
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Help message"
    )
    parser.add_argument(
        "-u", "--usages", action="help", default=argparse.SUPPRESS, help="Help message"
    )

    args = parser.parse_args()
    app = VolumeRenderingApp(
        render_config_file=str(args.config),
        density_volume_file=str(args.density),
        mask_volume_file=str(args.mask),
    )
    app.config()

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))
