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

from holoscan.core import Fragment, Operator, OperatorSpec
from holoscan.resources import Allocator, CudaStreamPool, UnboundedAllocator

from operators.grpc_operators.python.common.conditional_variable_queue import ConditionVariableQueue
from operators.grpc_operators.python.common.tensor_proto import TensorProto


class GrpcServerRequestOp(Operator):
    def __init__(
        self,
        fragment: Fragment,
        request_queue: ConditionVariableQueue,
        allocator: Allocator,
        cuda_stream_pool: CudaStreamPool,
        rpc_timeout: int = 5000,
        *args,
        **kwargs,
    ):
        self.request_queue: ConditionVariableQueue = request_queue
        self._allocator: Allocator = allocator
        self._cuda_stream_pool: CudaStreamPool = cuda_stream_pool
        self._rpc_timeout: int = rpc_timeout
        self.logger: logging.Logger = logging.getLogger(__name__)

        if not isinstance(request_queue, ConditionVariableQueue):
            raise ValueError("request_queue must be a ConditionVariableQueue")

        super().__init__(fragment, *args, **kwargs)

    @property
    def allocator(self) -> UnboundedAllocator:
        return self._allocator

    @property
    def rpc_timeout(self) -> int:
        return self._rpc_timeout

    @property
    def cuda_stream_pool(self) -> CudaStreamPool:
        return self._cuda_stream_pool

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if not self.request_queue.empty():
            request = self.request_queue.pop()

            entity = TensorProto.proto_to_tensor(request, context)
            op_output.emit(entity, "output")
