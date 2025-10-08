# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from florence2_op import Florence2Operator
from florence2_postprocessor_op import DetectionPostprocessorOp
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator


class FlorenceApp(Application):
    def set_parameters(self, task, prompt):
        """Set parameters for the Florence2Operator."""
        if self.florence_op:
            self.florence_op.task = task
            self.florence_op.prompt = prompt

    def compose(self):
        """Compose the application graph with various operators."""
        self.isprocessing = True

        # V4L2 to capture USB camera input
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("source"),
        )

        # CUDA stream pool for format conversion
        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Operator to convert video format to tensor
        format_converter_vlm = FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgba8888",
            out_dtype="rgb888",
            cuda_stream_pool=formatter_cuda_stream_pool,
            pool=UnboundedAllocator(self, name="FormatConverter allocator"),
        )

        # CUDA stream pool for Holoviz
        holoviz_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Detection postprocessor operator
        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
        )

        # Florence2 operator for inference
        florence_op = Florence2Operator(
            self,
            name="florence_op",
            **self.kwargs("florence_op"),
        )
        self.florence_op = florence_op

        # Holoviz operator to visualize the output
        holoviz = HolovizOp(
            self,
            width=854,
            height=480,
            window_title="Florence-2 Output",
            allocator=UnboundedAllocator(self, name="pool"),
            name="holoviz",
            cuda_stream_pool=holoviz_cuda_stream_pool,
            **self.kwargs("holoviz"),
        )

        # Connect Standard Operators
        self.add_flow(source, format_converter_vlm, {("signal", "source_video")})
        self.add_flow(format_converter_vlm, florence_op, {("tensor", "video_stream")})

        # Connect Florence2 output to the detection postprocessor
        self.add_flow(
            florence_op,
            detection_postprocessor,
            {("output", "input"), ("video_frame", "video_frame")},
        )

        # Connect the postprocessed output to Holoviz
        self.add_flow(florence_op, holoviz, {("video_frame", "receivers")})
        self.add_flow(detection_postprocessor, holoviz, {("outputs", "receivers")})
        self.add_flow(detection_postprocessor, holoviz, {("output_specs", "input_specs")})


def main():
    # Load the configuration file and run the application
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    app = FlorenceApp()
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
