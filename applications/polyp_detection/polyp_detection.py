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

import collections
import contextlib
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict

import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator


class PolypDetPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context, scores_threshold=0.5):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        pred_logits = in_message.get("pred_logits")
        pred_boxes = in_message.get("pred_boxes")

        print(f"pred_logits: {pred_logits.shape}, pred_boxes: {pred_boxes.shape}")


class PolypDetectionApp(Application):
    def __init__(
        self,
        data,
        source="replayer",
        video_size=(1164, 1034),
        inference_size=(640, 640),
        model_path="",
    ):
        """Initialize the colonoscopy detection application

        Parameters
        ----------
        source : {"replayer"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "Polyp Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data
        self.video_size = video_size
        self.inference_size = inference_size
        self.model_path = model_path

    def compose(self):
        n_channels = 3
        bpp = 4  # bytes per pixel

        video_dir = os.path.join(self.sample_data_path)
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )

        width_preprocessor = self.video_size[0]
        height_preprocessor = self.video_size[1]
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            resize_width=self.inference_size[0],
            resize_height=self.inference_size[1],
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            model_path_map={"polyp_det": self.model_path},
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = PolypDetPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        self.add_flow(source, detection_preprocessor)
        self.add_flow(detection_preprocessor, detection_inference, {("tensor", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Polyp Detection demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="/colon_workspace/polyp_detection_data",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer"],
        default="replayer",
        help=("If 'replayer', replay a prerecorded video."),
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help=("Set the model path"),
    )
    parser.add_argument(
        "--video_size",
        default=(1164, 1034),
        help=("Set the video size"),
    )
    parser.add_argument(
        "--inference_size",
        default=(640, 640),
        help=("Set the inference size"),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "polyp_detection.yaml")
    else:
        config_file = args.config
    app = PolypDetectionApp(
        data=args.data,
        source=args.source,
        video_size=args.video_size,
        inference_size=args.inference_size,
        model_path=args.model,
    )
    app.config(config_file)
    app.run()
