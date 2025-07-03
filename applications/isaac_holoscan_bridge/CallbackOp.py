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

from typing import Callable

from holoscan.core import ConditionType, Operator, OperatorSpec


class CallbackOp(Operator):
    """A Holoscan operator that executes a callback function when input data is ready.

    This operator is designed to handle multiple inputs and execute a user-provided callback
    function whenever all specified inputs have data available.

    Attributes:
        _data_ready_callback (Callable): Function to call when all inputs have data
        _inputs (list[str]): List of input port names to receive data from
        _optional_inputs (list[str]): List of optional input port names to receive data from

    Args:
        fragment: The Holoscan fragment this operator belongs to
        inputs (list[str]): List of input port names to receive data from
        optional_inputs (list[str]): List of optional input port names to receive data from
        data_ready_callback (Callable): Function to call when all inputs have data
        *args: Additional positional arguments passed to the Operator base class
        **kwargs: Additional keyword arguments passed to the Operator base class
    """

    def __init__(
        self,
        fragment,
        data_ready_callback: Callable,
        inputs: list[str],
        optional_inputs: list[str] = [],
        *args,
        **kwargs,
    ):
        # Store the callback function and input port names
        self._data_ready_callback = data_ready_callback
        self._inputs = inputs
        self._optional_inputs = optional_inputs
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Configure the operator's input ports.

        Args:
            spec (OperatorSpec): The operator specification object used to define inputs/outputs
        """
        # Register each input port specified in the inputs list
        for input in self._inputs:
            spec.input(input)
        for input in self._optional_inputs:
            spec.input(input).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Process incoming data and execute the callback function.

        This method is called by the Holoscan framework when data is available.
        It collects data from all input ports and passes it to the callback function
        as a dictionary mapping input names to their corresponding data.

        Args:
            op_input: Input data interface
            op_output: Output data interface (unused in this operator)
            context: Execution context
        """
        # Create a dictionary to store data from all inputs
        data_dict = {}

        # Process each input port
        for input in self._inputs:
            # Receive the message from the input port
            message = op_input.receive(input)
            # Extract the data from the message, the message is a dictionary, we ignore the key and
            # only get the first value
            data = list(message.values())[0]
            # Store the data in the dictionary using the input port name as the key
            data_dict[input] = data

        # Process each optional input port
        for input in self._optional_inputs:
            message = op_input.receive(input)
            if message is not None:
                data = list(message.values())[0]
                data_dict[input] = data

        # Execute the callback function with the collected data
        self._data_ready_callback(data_dict)
