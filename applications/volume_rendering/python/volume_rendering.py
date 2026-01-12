# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import logging
import os
import pathlib

from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator

from holohub.volume_loader import VolumeLoaderOp
from holohub.volume_renderer import VolumeRendererOp

logger = logging.getLogger("volume_rendering")


class JsonLoaderOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("json")
        spec.param("file_names")

    def compute(self, op_input, op_output, context):
        output = []
        for file_name in self.file_names:
            with open(file_name) as f:
                output.append(json.load(f))

        op_output.emit(output, "json")


class VolumeRenderingApp(Application):
    def __init__(
        self,
        argv=None,
        *args,
        render_config_file,
        render_preset_files,
        write_config_file,
        density_volume_file,
        density_min,
        density_max,
        mask_volume_file,
        count,
        **kwargs,
    ):
        self._rendering_config = render_config_file
        self._render_preset_files = render_preset_files
        self._write_config_file = write_config_file
        self._density_volume_file = density_volume_file
        self._mask_volume_file = mask_volume_file
        self._density_min = density_min
        self._density_max = density_max
        self._count = count

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

        density_volume_loader = VolumeLoaderOp(
            self,
            # the loader will executed only once to load the volume
            CountCondition(self, count=1),
            name="density_volume_loader",
            file_name=self._density_volume_file,
            allocator=volume_allocator,
        )

        mask_volume_loader = None
        if self._mask_volume_file:
            mask_volume_loader = VolumeLoaderOp(
                self,
                # the loader will executed only once to load the volume
                CountCondition(self, count=1),
                file_name=self._mask_volume_file,
                name="mask_volume_loader",
                allocator=volume_allocator,
            )

        preset_loader = None
        if self._render_preset_files:
            preset_loader = JsonLoaderOp(
                self,
                # the loader will executed only once to load the presets
                CountCondition(self, count=1),
                file_names=self._render_preset_files,
                name="preset_loader",
                allocator=volume_allocator,
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

        visualizer = HolovizOp(
            self,
            # stop application after short duration when testing
            CountCondition(self, count=self._count),
            name="viz",
            window_title="Volume Rendering with ClaraViz",
            enable_camera_pose_output=True,
            cuda_stream_pool=cuda_stream_pool,
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

        if mask_volume_loader:
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

        if preset_loader:
            self.add_flow(preset_loader, volume_renderer, {("json", "merge_settings")})
            # Since the preset_loader is only triggered once we have to set the input condition of
            # the merge_settings ports to ConditionType.NONE.
            # Currently, there is no API to set the condition of the receivers so we have to do this
            # after connecting the ports
            input = volume_renderer.spec.inputs["merge_settings:0"]
            if not input:
                raise RuntimeError("Could not find `merge_settings:0` input")
            input.condition(ConditionType.NONE)

        self.add_flow(volume_renderer, visualizer, {("color_buffer_out", "receivers")})
        self.add_flow(visualizer, volume_renderer, {("camera_pose_output", "camera_pose")})


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


def main():
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
        "-p",
        "--preset",
        action="append",
        type=valid_existing_path,
        dest="render_preset_files",
        help="Name of the renderer JSON preset file to load. This will be merged into the settings"
        "loaded from the configuration file. Multiple presets can be specified.",
    )
    parser.add_argument(
        "-w",
        "--write_config",
        action="store",
        type=pathlib.Path,
        dest="write_config_file",
        help="Name of the renderer JSON configuration file to write to (default '')",
    )
    parser.add_argument(
        "-d",
        "--density",
        action="store",
        default=argparse.SUPPRESS,
        type=valid_existing_path,
        dest="density",
        help=f"Name of density volume file to load (default {density_volume_file_default})",
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
        "-m",
        "--mask",
        action="store",
        default=argparse.SUPPRESS,
        type=valid_existing_path,
        dest="mask",
        help=f"Name of mask volume file to load (default {mask_volume_file_default})",
    )
    parser.add_argument(
        "-n",
        "--count",
        action="store",
        type=int,
        default=-1,
        dest="count",
        help="Duration to run application (default '-1' for unlimited duration)",
    )
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Help message"
    )
    parser.add_argument(
        "-u", "--usages", action="help", default=argparse.SUPPRESS, help="Help message"
    )

    args = parser.parse_args()

    if "density" not in args:
        args.density = density_volume_file_default
        args.mask = mask_volume_file_default

    app = VolumeRenderingApp(
        render_config_file=str(args.config),
        render_preset_files=(
            map(lambda x: str(x), args.render_preset_files) if args.render_preset_files else None
        ),
        write_config_file=str(args.write_config_file),
        density_volume_file=str(args.density),
        density_min=args.density_min,
        density_max=args.density_max,
        mask_volume_file=str(args.mask) if "mask" in args else None,
        count=args.count,
    )
    app.config()

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))

    print("Application has finished running.")


if __name__ == "__main__":
    main()
