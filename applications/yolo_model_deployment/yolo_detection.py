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

import os
from argparse import ArgumentParser

import cupy as cp
import numpy as np
import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    VideoStreamReplayerOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import UnboundedAllocator


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.
    Following the example of tensor_interop.py and ping.py4

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, width=640, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        # Convert input to numpy array (using CuPy)
        bboxes = cp.asarray(
            in_message.get("inference_output_detection_boxes")
        ).get()  # (nbatch, nboxes, ncoord)
        scores = cp.asarray(
            in_message.get("inference_output_detection_scores")
        ).get()  # (nbatch, nboxes)

        ix = scores.flatten() > 0  # Find the bbox with score >0
        if np.all(ix == False):
            bboxes = np.zeros([1, 2, 2], dtype=np.float32)
            scores = np.zeros([1, 1], dtype=np.float32)
        else:
            bboxes = bboxes[:, ix, :]
            scores = scores[:, ix]
            # Make box shape compatible with Holoviz
            bboxes = np.reshape(bboxes, (1, -1, 2))  # (nbatch, nboxes*2, ncoord/2)
            bboxes = bboxes / self.width  # The x, y need to be rescaled to [0,1]

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(bboxes), "rectangles")
        op_output.emit(out_message, "out")


class YoloDetApp(Application):
    """
    YOLO Detection Application.

    This application performs object detection using a YOLO model. It supports
    video input from a replayer or a V4L2 device and visualizes the detection results.

    Parameters:
        video_dir (str): Path to the video directory.
        data (str): Path to the model data directory.
        source (str): Input source, either "replayer" or "v4l2".
    """

    def __init__(self, video_dir, data, source="replayer"):
        super().__init__()
        self.name = "YOLO Detection App"
        self.source = source

        # Set default paths if not provided
        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "../data"), "yolo_model_deployment"
            )
        self.data = data

        if video_dir == "none":
            video_dir = data
        self.video_dir = video_dir

    def compose(self):
        # Resource allocator
        pool = UnboundedAllocator(self, name="pool")

        # Input source
        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"
            in_dtype = "rgba8888"  # V4L2 outputs RGBA8888
        else:
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.video_dir,
                **self.kwargs("replayer"),
            )
            source_output = "output"
            in_dtype = "rgb888"


        # Operators
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("detection_preprocessor"),
        )

        inference_kwargs = self.kwargs("detection_inference")
        for k, v in inference_kwargs["model_path_map"].items():
            inference_kwargs["model_path_map"][k] = os.path.join(self.data, v)

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="allocator"),
            **inference_kwargs,
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                ),
            ],
            **self.kwargs("detection_visualizer"),
        )

        # Data flow
        self.add_flow(source, detection_visualizer, {(source_output, "receivers")})
        self.add_flow(source, detection_preprocessor)
        self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "")})
        self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="YOLO Detection Demo Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=(
            "Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer."
        ),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help="Path to the model data directory.",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="none",
        help="Path to the video directory.",
    )

    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "yolo_detection.yaml")
    app = YoloDetApp(video_dir=args.video_dir, data=args.data, source=args.source)
    app.config(config_file)
    app.run()
