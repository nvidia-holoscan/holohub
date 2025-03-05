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

from queue import Queue
from typing import Optional

from holoscan.core import Application
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler

from holohub.grpc_operators import holoscan_pb2
from operators.grpc_operators.python.common.conditional_variable_queue import ConditionVariableQueue
from operators.grpc_operators.python.server.grpc_server_request import GrpcServerRequestOp
from operators.grpc_operators.python.server.grpc_server_response import GrpcServerResponseOp


class HoloscanGrpcApplication(Application):
    @property
    def composed(self) -> bool:
        return self._composed

    @composed.setter
    def composed(self, value: bool):
        self._composed = value

    @property
    def data_path(self) -> str:
        return self._data_path

    @data_path.setter
    def data_path(self, path: str):
        self._data_path = path

    @property
    def rpc_timeout(self) -> int:
        return self._grpc_request_op.rpc_timeout

    @property
    def grpc_request_op(self) -> GrpcServerRequestOp:
        return self._grpc_request_op

    @property
    def grpc_response_op(self) -> GrpcServerResponseOp:
        return self._grpc_response_op

    @property
    def request_queue(self) -> ConditionVariableQueue:
        return self._request_queue

    @property
    def response_queue(self) -> ConditionVariableQueue:
        return self._response_queue

    def __init__(self, incoming_request_queue: Queue, outgoing_response_queue: Queue):
        super().__init__()
        self.name: str = "gRPC Application"
        self._grpc_request_op: Optional[GrpcServerRequestOp] = None
        self._grpc_response_op: Optional[GrpcServerResponseOp] = None
        self._composed: bool = False
        self._request_queue: ConditionVariableQueue = ConditionVariableQueue(
            self, name="incoming_request_queue", queue=incoming_request_queue
        )
        self._response_queue = ConditionVariableQueue(
            self, name="outgoing_response_queue", queue=outgoing_response_queue
        )
        self.cuda_stream_pool: CudaStreamPool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        self.data_path = ""

    def compose(self):
        self._grpc_request_op = GrpcServerRequestOp(
            self,
            name="grpc_request_op",
            request_queue=self.request_queue,
            allocator=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=self.cuda_stream_pool,
            **self.kwargs("grpc_server"),
        )
        self._grpc_response_op = GrpcServerResponseOp(
            self,
            name="grpc_response_op",
            response_queue=self.response_queue,
        )

    def set_scheduler(self, config_name: str):
        self.scheduler(EventBasedScheduler(self, "scheduler", **self.kwargs(config_name)))

    def enqueue_request(self, request: holoscan_pb2.EntityRequest):
        self.request_queue.push(request)

    def enqueue_response(self, response: holoscan_pb2.EntityResponse):
        self.response_queue.push(response)

    def is_response_available(self) -> bool:
        return not self.response_queue.empty()

    def dequeue_response(self) -> holoscan_pb2.EntityResponse:
        return self.response_queue.pop()
