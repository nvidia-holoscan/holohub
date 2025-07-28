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

from argparse import ArgumentParser

import cupy as cp
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator
from holoscan.operators.holoviz import HolovizOp

from holohub.gamma_correction import GammaCorrectionOp


class SourceOp(Operator):
    """
    This operator serves as a data source in the Holoscan pipeline, emitting
    a 64x64x3 image with smooth color transitions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.shape = None

    def setup(self, spec):
        spec.output("output")

    def initialize(self):
        # Create a 64x64x3 RGB image with smooth color transitions
        width, height = 64, 64
        self.shape = (height, width, 3)

        # Create the image data on CPU first, then transfer to GPU
        self.data = np.zeros(self.shape, dtype=np.uint8)

        # Create an RGB image with smooth color transitions
        for y in range(height):
            for x in range(width):
                rgb = [0.0, 0.0, 0.0]

                # Red component: varies with x position
                rgb[0] = float(x) / width
                # Green component: varies with y position
                rgb[1] = float(y) / height
                # Blue component: inverse of x position
                rgb[2] = 1.0 - (float(x) / width)

                # Convert to uint8 and store
                for component in range(3):
                    self.data[y, x, component] = int((rgb[component] * 255.0) + 0.5)

    def compute(self, op_input, op_output, context):
        # Emit the pre-generated image data
        op_output.emit(dict(output=cp.asarray(self.data)), "output")


# @brief Application class for demonstrating gamma correction using SLANG shaders
#
# This application creates a workflow that:
# 1. Generates a test image with smooth color transitions
# 2. Applies gamma correction using a SLANG shader
# 3. Displays the result in HoloViz with proper sRGB color space handling
class SlangGammaCorrectionApp(Application):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def compose(self):
        source = SourceOp(self, name="Source")
        source.add_arg(CountCondition(self, self.count))

        gamma_correction = GammaCorrectionOp(
            self, name="GammaCorrection", data_type="uint8_t", component_count=3
        )

        # By default the image format is auto detected. Auto detection assumes linear color space,
        # but we provide an sRGB encoded image. Create an input spec and change the image format to
        # sRGB.
        input_spec = HolovizOp.InputSpec("output", HolovizOp.InputType.COLOR)
        input_spec.image_format = HolovizOp.ImageFormat.R8G8B8_SRGB

        holoviz = HolovizOp(
            self,
            name="Holoviz",
            window_title="Gamma Correction",
            framebuffer_srgb=True,
            tensors=[input_spec],
        )

        self.add_flow(source, gamma_correction, {("output", "input")})
        self.add_flow(gamma_correction, holoviz, {("output", "receivers")})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--count", type=int, default=-1)
    args = parser.parse_args()

    app = SlangGammaCorrectionApp(args.count)
    app.run()
