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

from holoscan.core import Operator, OperatorSpec

import hololink.sensors.csi
import cupy as cp


class RGBToBayerOp(Operator):
    """An operator that converts RGB to Bayer.

    This operator receives input data and converts it to Bayer format.

    Args:
        fragment: The fragment this operator belongs to
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Configure the operator's parameters, input and output specifications.

        Args:
            spec (OperatorSpec): The operator specification
        """
        spec.param("bayer_format", hololink.sensors.csi.BayerFormat.RGGB)
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Process the input data by calling the data ready callback.

        Args:
            op_input: The input port container
            op_output: The output port container
            context: The execution context
        """
        message = op_input.receive("in")
        data = cp.asarray(message.get(""))
        R = data[:, :, 0]
        G = data[:, :, 1]
        B = data[:, :, 2]

        bayer = cp.empty((data.shape[0], data.shape[1]), dtype=cp.uint8)
        if self.bayer_format == hololink.sensors.csi.BayerFormat.RGGB:
            # Convert RGB to RGGB Bayer pattern
            # Pattern:
            #    X0 X1
            # Y0 R  G
            # Y1 G  B
            bayer[0::2, 0::2] = R[0::2, 0::2]  # R (X0, Y0)
            bayer[0::2, 1::2] = G[0::2, 1::2]  # G (X1, Y0)
            bayer[1::2, 0::2] = G[1::2, 0::2]  # G (X0, Y1)
            bayer[1::2, 1::2] = B[1::2, 1::2]  # B (X1, Y1)
        else:
            raise ValueError(f"Unsupported Bayer format: {self.bayer_format}")
        op_output.emit({"": bayer}, "out")
