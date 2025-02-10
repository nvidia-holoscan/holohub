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

import asyncio
import logging

from holoscan.core import Operator, OperatorSpec

from operators.grpc_operators.python.common.tensor_proto import TensorProto


class GrpcClientRequestOp(Operator):
    @property
    def cuda_stream_pool(self):
        return self.cuda_stream_pool

    def __init__(self, fragment, request_queue, allocator, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.request_queue = request_queue
        self.allocator = allocator
        self.frame_count = 0

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("input")

        if not input:
            self.logger.warning("grpc: Failed to receive input message")
            return

        entity_request = TensorProto.tensor_to_proto(input)
        entity_request.frame_no = self.frame_count
        self.frame_count += 1
        asyncio.run(self.request_queue.push(entity_request))
        self.logger.debug(
            f"grpc: request queued for processing. #{entity_request.frame_no}, end={entity_request.end_of_stream}"
        )
