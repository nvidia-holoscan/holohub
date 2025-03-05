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

from holoscan.conditions import AsynchronousCondition, AsynchronousEventState
from holoscan.core import Fragment, Operator, OperatorSpec

from holohub.grpc_operators import holoscan_pb2
from operators.grpc_operators.python.common.asynchronous_condition_queue import (
    AsynchronousConditionQueue,
)
from operators.grpc_operators.python.common.tensor_proto import TensorProto


class GrpcClientResponseOp(Operator):
    def __init__(
        self,
        fragment: Fragment,
        response_queue: AsynchronousConditionQueue,
        condition: AsynchronousCondition,
        *args,
        **kwargs,
    ):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.response_queue: AsynchronousConditionQueue = response_queue
        self.condition: AsynchronousCondition = condition
        if not isinstance(condition, AsynchronousCondition):
            raise ValueError("condition must be a holoscan.condition.AsynchronousCondition")
        if not isinstance(response_queue, AsynchronousConditionQueue):
            raise ValueError("response_queue must be a AsynchronousConditionQueue")

        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        self.add_arg(self.condition)
        Operator.initialize(self)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def stop(self):
        self.condition.event_state = AsynchronousEventState.EVENT_NEVER

    def compute(self, op_input, op_output, context):
        entity_response: holoscan_pb2.EntityResponse = self.response_queue.pop()

        if entity_response:
            gxf_entity = TensorProto.proto_to_tensor(entity_response, context)
            op_output.emit(gxf_entity, "output")

        self.condition.event_state = AsynchronousEventState.EVENT_WAITING
