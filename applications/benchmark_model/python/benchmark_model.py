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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator


class App(Application):
    def __init__(self, data):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        """

        super().__init__()

        # set name
        self.name = "Benchmark Model App"

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")

        self.sample_data_path = data

        self.model_path = os.path.join(self.sample_data_path, "multiai_ultrasound")
        self.model_path_map = {
            "own_model": os.path.join(self.model_path, "aortic_stenosis.onnx"),
        }

        self.video_dir = os.path.join(self.sample_data_path, "multiai_ultrasound")
        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {self.video_dir=}")

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        source = VideoStreamReplayerOp(
            self, name="replayer", directory=self.video_dir, **self.kwargs("replayer")
        )

        preprocessor = FormatConverterOp(
            self, name="preprocessor", pool=host_allocator, **self.kwargs("preprocessor")
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.model_path_map,
            **self.kwargs("inference"),
        )

        postprocessor = SegmentationPostprocessorOp(
            self, name="postprocessor", allocator=host_allocator, **self.kwargs("postprocessor")
        )

        viz = HolovizOp(self, name="viz", **self.kwargs("viz"))

        # Define the workflow
        self.add_flow(source, viz, {("output", "receivers")})
        self.add_flow(source, preprocessor, {("output", "source_video")})
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in_tensor")})
        self.add_flow(postprocessor, viz, {("out_tensor", "receivers")})


def main(config_file, data):
    app = App(data=data)
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(
        description="Benchmark Model application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    default_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=default_data_path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="identity_model.onnx",
        help="Path to the model directory",
    )
    parser.add_argument(
        "-v", "--video-name", type=str, default="video", help="Path to the video file"
    )
    parser.add_argument(
        "-i",
        "--only-inference",
        action="store_true",
        help="Only run inference, no post-processing or visualization",
    )
    parser.add_argument(
        "-p",
        "--inference-postprocessing",
        action="store_true",
        help="Run inference and post-processing, no visualization",
    )
    parser.add_argument(
        "-l",
        "--multi-inference",
        type=int,
        default=1,
        help="Number of inferences to run in parallel",
    )
    # add positional argument CONFIG which is just a string
    config_file = os.path.join(os.path.dirname(__file__), "benchmark_model.yaml")
    parser.add_argument("ConfigPath", nargs="?", default=config_file, help="Path to config file")

    args = parser.parse_args()
    main(config_file=config_file, data=args.data)
