# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import argparse
import logging
import os

import cv2
import holoscan
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator
from skimage.io import imread

from holohub.apriltag_detector import ApriltagDetectorOp


def perspective_transform_usb(corners, width, height):
    top_l, top_r, bottom_r, bottom_l = corners
    dimensions = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32"
    )

    # Convert to Numpy format
    corners = np.array(corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dimensions)
    np.save("usb-cali.npy", matrix)
    print(f"matrix: {matrix}")


class AddBackgroundViewOperator(holoscan.core.Operator):
    def __init__(self, *args, width=1920, height=1080, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        self.width = width
        self.height = height

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("outputs")
        spec.output("output_specs")

    def start(self):
        self.background = imread("apriltag-calibration.png")

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message
        input_corners = op_input.receive("input")
        corners_list = np.zeros((4, 2), dtype=np.float32)
        for tag in input_corners:
            corners_list[tag.id][0] = tag.corners[tag.id][0]
            corners_list[tag.id][1] = tag.corners[tag.id][1]

        # Wait for 200 frames before calibration as Auto Focus might take
        # time to adjust the focus.
        if self.frame_count == 200:
            # perspective_transform_usb will write the calibration file to disk.
            perspective_transform_usb(corners_list, self.width, self.height)
        self.frame_count += 1
        out_message = {
            "image": self.background,
        }

        op_output.emit(out_message, "outputs")

        specs = []
        spec = HolovizOp.InputSpec("image", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.0
        view.height = 1.0
        spec.views = [view]
        specs.append(spec)
        op_output.emit(specs, "output_specs")


class UsbCamCalibrationApplication(holoscan.core.Application):
    def __init__(
        self,
        fullscreen,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._fullscreen = fullscreen
        self._frame_limit = frame_limit

    def compose(self):
        logging.info("compose")

        # USB Pipeline
        width = 1920
        height = 1080
        n_channels = 4
        block_size = width * height * n_channels
        allocator = holoscan.resources.BlockMemoryPool(
            self, name="pool", storage_type=0, block_size=block_size, num_blocks=1
        )
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=allocator,
            device="/dev/video0",
            width=width,
            height=height,
            gain=2,
        )
        source.add_arg(CountCondition(self, count=350))
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            out_tensor_name="preprocessed",
            in_dtype="rgba8888",
            out_dtype="rgb888",
            pool=UnboundedAllocator(self, name="pool_1"),
            # **preprocessor_args,
        )
        apriltag = ApriltagDetectorOp(
            self,
            name="apriltag",
            width=width,
            height=height,
            number_of_tags=4,
        )
        back_view = AddBackgroundViewOperator(
            self,
            name="back_view",
            width=width,
            height=height,
        )
        visualizer = HolovizOp(
            self,
            name="holoviz",
            fullscreen=True,
            **self.kwargs("holoviz"),
        )
        #
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, apriltag, {("", "input")})
        self.add_flow(apriltag, back_view, {("output", "input")})
        self.add_flow(back_view, visualizer, {("outputs", "receivers")})
        self.add_flow(back_view, visualizer, {("output_specs", "input_specs")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(os.path.dirname(__file__), "example_configuration.yaml")
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    args = parser.parse_args()

    # Set up the application
    application = UsbCamCalibrationApplication(
        args.fullscreen,
        args.frame_limit,
    )
    application.config(args.configuration)
    # Run it.
    application.run()


if __name__ == "__main__":
    main()
