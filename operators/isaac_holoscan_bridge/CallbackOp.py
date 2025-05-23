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
from typing import Callable


class CallbackOp(Operator):
    """An operator that executes a callback when data is received.

    This operator receives input data and calls a specified callback function with the
    received data.

    Args:
        fragment: The fragment this operator belongs to
        data_ready_callback (Callable): Callback function to be called when data is received
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(self, fragment, data_ready_callback: Callable, *args, **kwargs):
        self._data_ready_callback = data_ready_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Configure the operator's input and output specifications.

        Args:
            spec (OperatorSpec): The operator specification
        """
        spec.input("in")

    def compute(self, op_input, op_output, context):
        """Process the input data by calling the data ready callback.

        Args:
            op_input: The input port container
            op_output: The output port container
            context: The execution context
        """
        message = op_input.receive("in")
        data = message.get("")
        self._data_ready_callback(data)
