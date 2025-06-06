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

import random
import time

import numpy as np
from holoscan.core import Operator, OperatorSpec


class ControlOp(Operator):
    """A Holoscan operator that controls joint positions by generating random movements.

    This operator receives joint positions as input and periodically (every 2 seconds)
    generates new random joint positions within a range of [-1.5, 1.5]. It's designed
    for testing and demonstration purposes of robotic arm control.

    Attributes:
        _next_move (float): Timestamp for when the next random movement should occur.
    """

    def __init__(self, fragment, *args, **kwargs):
        """Initialize the ControlOp operator.

        Args:
            fragment: The Holoscan fragment this operator belongs to.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._next_move = time.time()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Configure the operator's input and output ports.

        Args:
            spec (OperatorSpec): The operator specification object used to define
                input and output ports.
        """
        spec.input("input_joint_positions")
        spec.output("output_joint_positions")

    def compute(self, op_input, op_output, context):
        """Process input data and generate random joint positions.

        This method is called on each tick of the Holoscan pipeline. It checks if it's
        time to generate new random positions (every 2 seconds) and if so, generates
        random values for each joint within the range [-1.5, 1.5].

        Args:
            op_input: The input data containing current joint positions.
            op_output: The output object to emit new joint positions.
            context: The execution context.
        """
        message = op_input.receive("input_joint_positions")

        # move the arm to a random position every two seconds
        if time.time() > self._next_move:
            data = np.asarray(list(message.values())[0])

            for i in range(data.shape[1]):
                data[0][i] = random.uniform(-1.5, 1.5)

            self._next_move = time.time() + 2.0

            op_output.emit({"joint_positions": data}, "output_joint_positions")
