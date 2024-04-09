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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from holoscan.core import (
    Application,
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
)
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    V4L2VideoCaptureOp,
)
from holoscan.resources import UnboundedAllocator


class SinkOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        _ = op_input.receive("in")


class App(Application):
    def __init__(
        self,
        datapath,
        model_name,
        num_inferences,
        only_inference,
        inference_postprocessing,
    ):
        """Initialize the application"""

        super().__init__()

        # set name
        self.name = "Benchmark Model App"

        self.datapath = datapath
        self.model_name = model_name
        self.num_inferences = num_inferences
        self.only_inference = only_inference
        self.inference_postprocessing = inference_postprocessing

        if not os.path.exists(self.datapath):
            raise ValueError(f"Data path {self.datapath} does not exist.")

        self.model_path = os.path.join(self.datapath, model_name)
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path {self.model_path} does not exist.")

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=host_allocator,
            **self.kwargs("source"),
        )

        preprocessor = FormatConverterOp(
            self, name="preprocessor", pool=host_allocator, **self.kwargs("preprocessor")
        )

        model_path_map = {}
        pre_processor_map = {}
        inference_map = {}

        for i in range(0, self.num_inferences):
            model_path_map[f"own_model_{i}"] = self.model_path
            pre_processor_map[f"own_model_{i}"] = ["source_video"]
            inference_map[f"own_model_{i}"] = [f"output{i}"]

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=model_path_map,
            pre_processor_map=pre_processor_map,
            inference_map=inference_map,
            **self.kwargs("inference"),
        )

        holovizs = []
        if not self.only_inference and not self.inference_postprocessing:
            for i in range(0, self.num_inferences):
                viz = HolovizOp(self, name=f"holoviz{i}", **self.kwargs("viz"))
                holovizs.append(viz)
                # Passthrough to Visualization
                self.add_flow(source, viz, {("signal", "receivers")})

        # Inference path
        self.add_flow(source, preprocessor, {("signal", "source_video")})
        self.add_flow(preprocessor, inference, {("tensor", "receivers")})

        if self.only_inference:
            print("Only inference mode is on, no post-processing and visualization will be done.")
            sink = SinkOp(self, name="sink")
            self.add_flow(inference, sink)
            return

        postprocessors = []
        for i in range(0, self.num_inferences):
            in_tensor_name = f"output{i}"
            postprocessor = SegmentationPostprocessorOp(
                self,
                name=f"postprocessor{i}",
                allocator=host_allocator,
                in_tensor_name=in_tensor_name,
                **self.kwargs("postprocessor"),
            )
            postprocessors.append(postprocessor)
            self.add_flow(inference, postprocessor, {("transmitter", "in_tensor")})

        if self.inference_postprocessing:
            print("Inference and post-processing mode is on, no visualization will be done.")
            for i in range(0, self.num_inferences):
                sink = SinkOp(self, name=f"sink{i}")
                self.add_flow(postprocessors[i], sink)
            return

        for i in range(0, self.num_inferences):
            self.add_flow(postprocessors[i], holovizs[i], {("out_tensor", "receivers")})


def main(args):
    app = App(
        args.data,
        args.model_name,
        args.multi_inference,
        args.only_inference,
        args.inference_postprocessing,
    )
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(
        description="Benchmark Model Application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    default_data_path = "/workspace/holohub/data/ultrasound_segmentation"
    default_model_name = "us_unet_256x256_nhwc.onnx"
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
        default=default_model_name,
        help="Path to the model directory",
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
    config_file = os.path.join(os.path.dirname(__file__), "model_benchmarking.yaml")
    parser.add_argument("ConfigPath", nargs="?", default=config_file, help="Path to config file")

    args = parser.parse_args()
    main(args)
