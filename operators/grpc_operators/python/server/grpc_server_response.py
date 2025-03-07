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

import logging

from holoscan.core import IOSpec, Operator, OperatorSpec

from holohub.grpc_operators import holoscan_pb2 as holoscan_proto
from operators.grpc_operators.python.common.tensor_proto import TensorProto
from operators.grpc_operators.python.server.grpc_application import ConditionVariableQueue


class GrpcServerResponseOp(Operator):
    def __init__(self, fragment, response_queue: ConditionVariableQueue, *args, **kwargs):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.response_queue: ConditionVariableQueue = response_queue
        if not isinstance(response_queue, ConditionVariableQueue):
            raise ValueError("response_queue must be a ConditionVariableQueue")

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        input_messages = op_input.receive("input")
        response = holoscan_proto.EntityResponse()
        tensors = 0
        for message in input_messages:
            TensorProto.add_tensor_to_proto(message, response)
            tensors += 1

        if tensors > 0:
            self.response_queue.push(response)
