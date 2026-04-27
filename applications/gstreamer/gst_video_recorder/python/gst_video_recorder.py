# SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
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

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError(
        "gst_video_recorder Python implementation requires CuPy. "
        "Use the containerized build or install the matching CuPy wheel "
        "for your CUDA version (for example, cupy-cuda12x or cupy-cuda13x)."
    ) from exc

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator

from holohub.holoscan_gstreamer_bridge import GstVideoRecorderOp


def parse_pattern(value: str) -> int:
    try:
        pattern = int(str(value).strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "invalid --pattern; use 0 (gradient), 1 (checkerboard), or 2 (color bars)"
        ) from exc

    if pattern not in {0, 1, 2}:
        raise argparse.ArgumentTypeError(
            "invalid --pattern; use 0 (gradient), 1 (checkerboard), or 2 (color bars)"
        )
    return pattern


RGBA_CHANNELS = 4
ALPHA_OPAQUE = 255
CHECKERBOARD_BASE_SIZE = 64
CHECKERBOARD_VARIATION = 32
SMPTE_COLOR_BARS = 7
GRADIENT_TIME_STEP = 0.02
CHECKERBOARD_TIME_STEP = 0.05


def parse_v4l2_pixel_format(value: str) -> str:
    pixel_format = str(value).strip()
    if not pixel_format:
        raise argparse.ArgumentTypeError("pixel format cannot be empty")
    return pixel_format


def parse_key_value_properties(items: list[str]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --property '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit("property key cannot be empty")
        if not value:
            raise SystemExit("property value cannot be empty")
        props[key] = value
    return props


class PatternGeneratorOp(Operator):
    """Emit RGBA frames for pattern-generator mode matching the C++ implementation."""

    def __init__(
        self,
        fragment,
        *args,
        width: int = 1920,
        height: int = 1080,
        pattern: int = 0,
        storage_type: int = 1,
        **kwargs,
    ):
        self.width = int(width)
        self.height = int(height)
        self.pattern = int(pattern)
        self.storage_type = int(storage_type)
        self.xp = np if self.storage_type == 0 else cp

        self.time_offset = 0.0
        self.animation_time = 0.0
        self._x = None
        self._y = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self._y, self._x = self.xp.indices((self.height, self.width), dtype=self.xp.float32)

    def _gradient(self):
        assert self._x is not None and self._y is not None
        self.time_offset += GRADIENT_TIME_STEP

        frame = self.xp.empty((self.height, self.width, RGBA_CHANNELS), dtype=self.xp.uint8)
        frame[..., 0] = (128.0 + 127.0 * self.xp.sin(self._x * 0.01 + self.time_offset)).astype(
            self.xp.uint8
        )
        frame[..., 1] = (128.0 + 127.0 * self.xp.sin(self._y * 0.01 + self.time_offset)).astype(
            self.xp.uint8
        )
        frame[..., 2] = (
            128.0 + 127.0 * self.xp.cos((self._x + self._y) * 0.005 + self.time_offset)
        ).astype(self.xp.uint8)
        frame[..., 3] = ALPHA_OPAQUE
        return frame

    def _checkerboard(self):
        assert self._x is not None and self._y is not None
        self.animation_time += CHECKERBOARD_TIME_STEP

        square_size = CHECKERBOARD_BASE_SIZE + int(
            CHECKERBOARD_VARIATION * np.sin(self.animation_time)
        )
        square_size = max(1, square_size)

        board = (
            (self._x.astype(self.xp.int32) // square_size)
            + (self._y.astype(self.xp.int32) // square_size)
        ) % 2 == 0
        color = self.xp.where(board, ALPHA_OPAQUE, 0).astype(self.xp.uint8)

        frame = self.xp.empty((self.height, self.width, RGBA_CHANNELS), dtype=self.xp.uint8)
        frame[..., 0] = color
        frame[..., 1] = color
        frame[..., 2] = color
        frame[..., 3] = ALPHA_OPAQUE
        return frame

    def _colorbars(self):
        colors = self.xp.array(
            [
                [255, 255, 255, ALPHA_OPAQUE],  # White
                [255, 255, 0, ALPHA_OPAQUE],  # Yellow
                [0, 255, 255, ALPHA_OPAQUE],  # Cyan
                [0, 255, 0, ALPHA_OPAQUE],  # Green
                [255, 0, 255, ALPHA_OPAQUE],  # Magenta
                [255, 0, 0, ALPHA_OPAQUE],  # Red
                [0, 0, 255, ALPHA_OPAQUE],  # Blue
            ],
            dtype=self.xp.uint8,
        )

        frame = self.xp.empty((self.height, self.width, RGBA_CHANNELS), dtype=self.xp.uint8)
        bar_width = self.width // SMPTE_COLOR_BARS
        bar_width = max(1, bar_width)

        x_coords = self.xp.arange(self.width, dtype=self.xp.int32)
        bar_indices = x_coords // bar_width
        bar_indices = self.xp.minimum(bar_indices, SMPTE_COLOR_BARS - 1)

        column_colors = colors[bar_indices]
        frame[...] = column_colors[self.xp.newaxis, :, :]

        return frame

    def compute(self, op_input, op_output, context):
        if self.pattern == 0:
            frame = self._gradient()
        elif self.pattern == 1:
            frame = self._checkerboard()
        elif self.pattern == 2:
            frame = self._colorbars()
        else:
            frame = self._gradient()

        op_output.emit({"video_frame": frame}, "output")


class GstVideoRecorderApp(Application):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

    def _source_condition_args(self) -> list[Any]:
        if self.args.count > 0:
            return [CountCondition(self, self.args.count)]
        return []

    def compose(self):
        condition_args = self._source_condition_args()

        recorder = GstVideoRecorderOp(
            self,
            encoder=self.args.encoder,
            framerate=self.args.framerate,
            max_buffers=10,
            filename=self.args.output,
            properties=self.args.properties,
            name="gst_video_recorder",
        )

        if self.args.source == "pattern":
            source = PatternGeneratorOp(
                self,
                *condition_args,
                width=self.args.width,
                height=self.args.height,
                pattern=self.args.pattern,
                storage_type=self.args.storage,
                name="pattern_source",
            )
            self.add_flow(source, recorder, {("output", "input")})
        elif self.args.source == "v4l2":
            allocator = UnboundedAllocator(self, name="allocator")

            source = V4L2VideoCaptureOp(
                self,
                *condition_args,
                allocator=allocator,
                device=self.args.device,
                width=self.args.width,
                height=self.args.height,
                pixel_format=self.args.pixel_format,
                name="v4l2_source",
            )
            format_converter = FormatConverterOp(
                self,
                name="format_converter",
                in_dtype="rgba8888",
                out_dtype="rgba8888",
                pool=allocator,
            )

            self.add_flow(source, format_converter, {("signal", "source_video")})
            self.add_flow(format_converter, recorder, {("tensor", "input")})

        else:
            raise RuntimeError(f"unsupported source '{self.args.source}'")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Holohub Python sample for GstVideoRecorderOp.",
        add_help=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )

    # General options
    parser.add_argument(
        "--source",
        choices=("pattern", "v4l2"),
        default="pattern",
        help="input source type",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.mp4",
        help="output video filename",
    )
    parser.add_argument(
        "-e",
        "--encoder",
        default="nvh264",
        help="encoder base name (for example: nvh264, nvh265, x264, x265); the 'enc' suffix is added automatically by the recorder operator",
    )
    parser.add_argument(
        "-f",
        "--framerate",
        default="30/1",
        help='frame rate as fraction or decimal (for example "30/1", "30000/1001", "29.97", "60")',
    )
    parser.add_argument(
        "-c",
        "--count",
        dest="count",
        type=int,
        default=None,
        help="number of frames to produce or capture; 0 means unlimited (default: unlimited)",
    )
    parser.add_argument(
        "--property",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra GStreamer encoder property; may be repeated",
    )

    # Resolution options
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1920,
        help="frame width in pixels",
    )
    parser.add_argument(
        "-h",
        "--height",
        type=int,
        default=1080,
        help="frame height in pixels",
    )

    # V4L2 options via built-in Holoscan operator
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="V4L2 device path",
    )
    parser.add_argument(
        "--pixel-format",
        type=parse_v4l2_pixel_format,
        default="auto",
        help="V4L2 pixel format (for example: YUYV, MJPEG, auto)",
    )

    # Pattern generator options
    parser.add_argument(
        "--pattern",
        type=parse_pattern,
        default=0,
        help="pattern type: 0 = animated gradient, 1 = animated checkerboard, 2 = color bars (SMPTE style)",
    )
    parser.add_argument(
        "--storage",
        type=int,
        choices=(0, 1),
        default=1,
        help="memory storage type: 0 = host memory, 1 = device or CUDA memory (default: 1)",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.output).strip():
        raise SystemExit("--output cannot be empty")
    if not str(args.encoder).strip():
        raise SystemExit("--encoder cannot be empty")
    if args.source == "v4l2" and not str(args.device).strip():
        raise SystemExit("--device cannot be empty when --source is v4l2")
    if not (64 <= args.width <= 8192):
        raise SystemExit("--width must be between 64 and 8192")
    if not (64 <= args.height <= 8192):
        raise SystemExit("--height must be between 64 and 8192")
    if not args.framerate or not str(args.framerate).strip():
        raise SystemExit("--framerate cannot be empty")
    if args.count is None:
        args.count = 0
    elif not (0 <= args.count <= 1_000_000_000):
        raise SystemExit("--count must be 0 (unlimited) or between 1 and 1000000000")
    args.properties = parse_key_value_properties(args.property)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    try:
        app = GstVideoRecorderApp(args)
        app.run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
