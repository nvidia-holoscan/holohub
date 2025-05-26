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
    def __init__(self, fragment, inputs: list[str], data_ready_callback: Callable, *args, **kwargs):
        self._data_ready_callback = data_ready_callback
        self._inputs = inputs
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for input in self._inputs:
            spec.input(input)

    def compute(self, op_input, op_output, context):
        data_dict = {}
        for input in self._inputs:
            message = op_input.receive(input)
            data = message.get("")
            data_dict[input] = data
        self._data_ready_callback(data_dict)
