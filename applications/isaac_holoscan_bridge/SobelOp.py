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

import cupy as cp
import cupyx.scipy.ndimage
from holoscan.core import Operator, OperatorSpec


class SobelOp(Operator):
    """An operator that performs Sobel edge detection on an image.

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
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        """Process the input data by performing Sobel edge detection.

        Args:
            op_input: The input port container
            op_output: The output port container
            context: The execution context
        """
        message = op_input.receive("input")

        outputs = {}
        for key, data in message.items():
            outputs[f"{key}_sobel"] = cupyx.scipy.ndimage.sobel(cp.asarray(data), axis=0)

        op_output.emit(outputs, "output")
