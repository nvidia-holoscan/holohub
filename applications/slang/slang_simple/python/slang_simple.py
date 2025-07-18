# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import cupy as cp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator

from holohub.slang_shader import SlangShaderOp


class SourceOp(Operator):
    """
    A simple source operator that generates a single-element Tensor containing incrementing
    integer values.

    This operator serves as a data source in the Holoscan pipeline, emitting
    incrementing integer values starting from 1. Each compute cycle produces
    a new value that gets passed to downstream operators.

    Attributes:
        index (int): The current value to emit, increments after each emission
    """

    def __init__(self, *args, **kwargs):
        self.index = 1
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        value = cp.array([self.index], dtype=cp.int32)
        op_output.emit(dict(output=value), "output")
        self.index += 1


class SinkOp(Operator):
    """
    A simple sink operator that receives and prints data from upstream operators.

    This operator acts as a data sink in the Holoscan pipeline, receiving
    tensor data from upstream operators and printing the received values.
    It's typically used for debugging or final data consumption in a pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("input")
        _, tensor = tensormap.popitem()
        print(f"Received value: {cp.asarray(tensor)[0]}")


class SlangSimpleApp(Application):
    """
    A Holoscan application demonstrating Slang shader integration.

    This application creates a simple pipeline that:
    1. Generates incrementing integer values using SourceOp
    2. Processes the data through a Slang shader using SlangShaderOp
    3. Receives and prints the processed results using SinkOp

    The pipeline demonstrates how to integrate Slang shaders into Holoscan
    applications for GPU-accelerated data processing.

    Pipeline Flow:
        SourceOp -> SlangShaderOp -> SinkOp
    """

    def compose(self):
        source = SourceOp(self, name="Source")
        sink = SinkOp(self, name="Sink")

        slang = SlangShaderOp(self, name="Slang", shader_source_file="simple.slang")
        # Execute the pipeline10 times
        slang.add_arg(CountCondition(self, 10))

        # Set the value of a parameter defined in the Slang shader
        slang.add_arg(offset=10)

        self.add_flow(source, slang, {("output", "input_buffer")})
        self.add_flow(slang, sink, {("output_buffer", "input")})


if __name__ == "__main__":
    app = SlangSimpleApp()
    app.run()
