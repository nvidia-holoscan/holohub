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
from polyp_det_operator import PolypDetInferenceOp, PolypDetPostprocessorOp
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator


class PolypDetectionApp(Application):
    def __init__(self, data, source="replayer"):
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

    def compose(self):
        n_channels = 3
        bpp = 4  # bytes per pixel

        video_dir = os.path.join(self.sample_data_path)
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )

        width_preprocessor = 640
        height_preprocessor = 640
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = PolypDetInferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            orig_size=(1164, 1034),
            trt_model="/colon_workspace/RT-DETR/rtdetrv2_pytorch/rt_detrv2_timm_r50_nvimagenet_pretrained_neg_finetune.trt",
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = PolypDetPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        self.add_flow(source, detection_preprocessor)
        self.add_flow(detection_preprocessor, detection_inference, {("tensor", "input")})
        self.add_flow(detection_inference, detection_postprocessor, {("output", "in")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Colonoscopy segmentation demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer"],
        default="replayer",
        help=("If 'replayer', replay a prerecorded video."),
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
        default="/colon_workspace/holohub/data/polyp_detection",
        help=("Set the data path"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "polyp_detection.yaml")
    else:
        config_file = args.config
    app = PolypDetectionApp(data=args.data, source=args.source)
    app.config(config_file)
    app.run()
