# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from argparse import ArgumentParser

import cupy as cp
import cv2
import holoscan as hs
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator


class PostprocessorOp(Operator):
    """Operator that does postprocessing before sending resulting image to Holoviz"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.image_dim = 518
        self.mouse_pressed = False
        self.display_modes = ["original", "depth", "side-by-side", "interactive"]
        self.idx = 1
        self.current_display_mode = self.display_modes[self.idx]
        # In interactive mode, how much of the original video to show
        self.ratio = 0.5

    def setup(self, spec: OperatorSpec):
        """
        input:  "input_depthmap"  - Input tensors representing depthmap from inference
        input:  "input_image"     - Input tensor representing the RGB image
        output: "output_image"    - The image for Holoviz to display
        output: "output_specs"    - Text to show the current display mode

        This operator's output image depends on the current display mode, if set to

            * "original": output the original image from input source
            * "depth": output the color depthmap based on the depthmap returned from
                       Depth Anything V2 model
            * "side-by-side": output a side-by-side view of the original image next to
                              the color depthmap
            * "interactive": allow user to control how much of the image to show as
                             original while the rest shows the color depthmap

        Returns:
            None
        """
        spec.input("input_depthmap")
        spec.input("input_image")
        spec.output("output_image")
        spec.output("output_specs")

    def clamp(self, value, min_value=0, max_value=1):
        """Clamp value between [min_value, max_value]"""
        return max(min_value, min(max_value, value))

    def toggle_display_mode(self, *args):
        mouse_button = args[0]
        action = args[1]

        LEFT_BUTTON = 0
        PRESSED = 0

        # If event is for the middle or right mouse button, update some values for interactive mode
        #   - update the status of whether the button is being pressed or released
        #   - update the ratio of the original image to display
        if mouse_button.value != LEFT_BUTTON:
            self.mouse_pressed = action.value == PRESSED
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size
            return

        # When left mouse button is pressed, update the display mode
        if action.value == PRESSED:
            self.idx = (self.idx + 1) % len(self.display_modes)
            self.current_display_mode = self.display_modes[self.idx]

    # Update cursor position which will be used in interactive mode
    def cursor_pos_callback(self, *args):
        self.x = args[0]
        if self.mouse_pressed:
            self.x = self.clamp(self.x, 0, self.framebuffer_size)
            self.ratio = self.x / self.framebuffer_size

    # Update size of holoviz framer buffer which will be used to calculate self.ratio
    def framebuffer_size_callback(self, *args):
        self.framebuffer_size = args[0]

    def normalize(self, depth_map):
        min_value = cp.min(depth_map)
        max_value = cp.max(depth_map)
        normalized = (depth_map - min_value) / (max_value - min_value)
        return 255 - (normalized * 255)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input_depthmap")
        in_image = op_input.receive("input_image")

        # Convert input to cupy array
        inference_output = cp.asarray(in_message.get("inference_output")).squeeze()

        image = cp.asarray(in_image.get("preprocessed"))

        if self.current_display_mode == "original":
            # Display the original image
            image = (image * 255).astype(cp.uint8)
            output_image = image
        elif self.current_display_mode == "depth":
            # Display the color depthmap
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            output_image = depth_colormap
        elif self.current_display_mode == "side-by-side":
            # Display both original and color depthmap images side-by-side
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            output_image = cp.hstack((image, depth_colormap))
        else:
            # Interactive mode
            depth_normalized = self.normalize(inference_output)
            depth_colormap = cv2.applyColorMap(
                depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET
            )
            image = (image * 255).astype(cp.uint8)
            pos = int(self.image_dim * self.ratio)
            output_image = cp.hstack(
                (
                    image[:, :pos, :],
                    depth_colormap[
                        :,
                        pos:,
                    ],
                )
            )

        # Position display mode text near bottom left corner of Holoviz window
        display_mode_text = np.asarray([(0.025, 0.9)])

        # Create output message
        out_message = {"display_mode": display_mode_text, "image": hs.as_tensor(output_image)}
        op_output.emit(out_message, "output_image")

        # holoviz specs for displaying the current display mode
        specs = []
        spec = HolovizOp.InputSpec("display_mode", "text")
        spec.text = [self.current_display_mode]
        spec.color = [1.0, 1.0, 1.0, 1.0]
        spec.priority = 1
        specs.append(spec)
        op_output.emit(specs, "output_specs")


class DepthAnythingV2App(Application):
    def __init__(self, data, source="v4l2", video_device="none"):
        """Initialize the depth anything v2 application"""

        super().__init__()

        # set name
        self.name = "Depth Anything V2 App"
        self.source = source

        if data == "none":
            data = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "depth_anything_v2")

        self.sample_data_path = data
        self.video_device = video_device

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Input data type of preprocessor
        in_dtype = "rgb888"

        if self.source == "v4l2":
            v4l2_args = self.kwargs("v4l2_source")
            if self.video_device != "none":
                v4l2_args["device"] = self.video_device
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **v4l2_args,
            )
            # v4l2 operator outputs RGBA8888
            in_dtype = "rgba8888"
        elif self.source == "replayer":
            source = VideoStreamReplayerOp(
                self,
                name="replayer_source",
                directory=self.sample_data_path,
                **self.kwargs("replayer_source"),
            )

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **preprocessor_args,
        )

        inference_args = self.kwargs("inference")
        inference_args["model_path_map"] = {
            "depth": os.path.join(self.sample_data_path, "depth_anything_v2_vits.onnx")
        }

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_args,
        )

        postprocessor = PostprocessorOp(self, name="postprocessor", allocator=pool)

        holoviz_args = self.kwargs("holoviz")

        # Register mouse event callbacks
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="DepthAnything v2",
            mouse_button_callback=postprocessor.toggle_display_mode,
            cursor_pos_callback=postprocessor.cursor_pos_callback,
            framebuffer_size_callback=postprocessor.framebuffer_size_callback,
            **holoviz_args,
        )

        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, postprocessor, {("tensor", "input_image")})
        self.add_flow(preprocessor, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "input_depthmap")})
        self.add_flow(postprocessor, holoviz, {("output_image", "receivers")})
        self.add_flow(postprocessor, holoviz, {("output_specs", "input_specs")})


def main():
    # Parse args
    parser = ArgumentParser(description="Depth Anything V2 Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file or "
            " --video_device if specified. "
            "If 'replayer', uses video stream replayer."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-v",
        "--video_device",
        default="none",
        help=("The video device to use.  By default the application will use /dev/video0"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "depth_anything_v2.yaml")
    else:
        config_file = args.config

    app = DepthAnythingV2App(args.data, args.source, args.video_device)
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
