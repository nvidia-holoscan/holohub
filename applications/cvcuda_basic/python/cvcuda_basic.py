"""
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # no qa

import os
from argparse import ArgumentParser

import cvcuda
import nvcv
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp


# Define custom Operators for use in the demo
class ImageProcessingOp(Operator):
    """Example of an operator processing input video (as a tensor).

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"

    Each input frame is processed by CV-CUDA's flip operator to perform
    a vertical flip of the video.

    In this demo, the input and output image (2D RGB) is a 3D array of shape
    (height, width, channels).
    """

    def __init__(self, fragment, *args, use_flip_into=False, **kwargs):
        self.cv_out = None

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("input_tensor")
        input_tensor = tensormap[""]  # stride (2562, 3, 1)

        cv_in = nvcv.as_tensor(input_tensor, "HWC")  # Input_tensor is (480, 854, 3)

        if self.cv_out is None:
            # store output buffer for future reuse
            self.cv_out = cvcuda.flip(src=cv_in, flipCode=0)
        else:
            # reuse the buffer from the first compute call
            cvcuda.flip_into(src=cv_in, dst=self.cv_out, flipCode=0)

        buffer = self.cv_out.cuda()
        output_tensormap = dict(image=buffer)
        op_output.emit(output_tensormap, "output_tensor")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - ImageProcessingOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
    The ImageProcessingOp processes the frames and sends the processed frames to the HolovizOp.
    The HolovizOp displays the processed frames.
    """

    def __init__(self, *args, data, count=0, **kwargs):
        super().__init__(*args, **kwargs)

        if data == "none":
            data = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "endoscopy")

        self.sample_data_path = data
        self.count = count

    def compose(self):
        width = 854
        height = 480
        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="surgical_video",
            frame_rate=0,
            repeat=True,
            realtime=True,
            count=self.count,
        )

        image_processing = ImageProcessingOp(self, name="image_processing")

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[dict(name="image", type="color", opacity=1.0, priority=0)],
        )

        self.add_flow(source, image_processing)
        self.add_flow(image_processing, visualizer, {("output_tensor", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="CV-CUDA demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=0,
        help=("Number of frames to play (0 = run until user closes the window)"),
    )
    args = parser.parse_args()
    app = MyVideoProcessingApp(data=args.data, count=args.count)
    app.run()
