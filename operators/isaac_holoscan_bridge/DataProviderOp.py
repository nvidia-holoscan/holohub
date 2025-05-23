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

from holoscan.conditions import PeriodicCondition
from holoscan.core import Operator, OperatorSpec

import datetime
import cupy as cp


class DataProviderOp(Operator):
    """A Holoscan operator that provides data by generating sequential buffer values.

    This operator generates a sequence of buffers filled with incrementing values.
    Each buffer is created with the specified size and emitted as output.

    Args:
        fragment: The Holoscan fragment this operator belongs to.
        buffer_size (int): The size of the buffer to generate.
        *args: Additional positional arguments passed to the parent Operator.
        **kwargs: Additional keyword arguments passed to the parent Operator.
    """

    def __init__(self, fragment, *args, buffer_size: int, fps: float, **kwargs):
        """Initialize the DataProviderOp.

        Args:
            fragment: The Holoscan fragment this operator belongs to.
            buffer_size (int): The size of the buffer to generate.
            *args: Additional positional arguments passed to the parent Operator.
            **kwargs: Additional keyword arguments passed to the parent Operator.
        """
        self._value = 0
        self._buffer_size = buffer_size
        self._fps = fps
        self._condition = PeriodicCondition(
            fragment, recess_period=datetime.timedelta(seconds=1.0 / self._fps)
        )
        super().__init__(fragment, self._condition, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the operator specification.

        Args:
            spec (OperatorSpec): The operator specification to configure.
        """
        spec.output("out")

    def compute(self, op_input, op_output, context):
        """Compute the next buffer of data.

        This method generates a new buffer filled with the current value and increments
        the value for the next call. The buffer is emitted as output.

        Args:
            op_input: The operator input (not used in this operator).
            op_output: The operator output where the generated buffer is emitted.
            context: The operator context (not used in this operator).
        """
        data = cp.full(self._buffer_size, self._value, dtype=cp.uint8)
        self._value += 1
        op_output.emit({"": data}, "out")
