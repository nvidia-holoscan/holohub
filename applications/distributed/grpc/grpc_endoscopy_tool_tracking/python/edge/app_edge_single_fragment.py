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

from holoscan.conditions import AsynchronousCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType, RMMAllocator

from operators.grpc_operators.python.client.entity_client_service import EntityClientService
from operators.grpc_operators.python.client.grpc_client_request import GrpcClientRequestOp
from operators.grpc_operators.python.client.grpc_client_response import GrpcClientResponseOp
from operators.grpc_operators.python.common.asynchronous_condition_queue import (
    AsynchronousConditionQueue,
)
from operators.grpc_operators.python.common.asyncio_queue import AsyncIoQueue


class AppEdgeSingleFragment(Application):
    def __init__(self, data_path: str):
        self.logger = logging.getLogger(__name__)
        self.datapath = data_path
        self.entity_client_service = None

        super().__init__()

    async def cleanup(self):
        """Async cleanup method"""
        if self.entity_client_service:
            try:
                # Shield the stop operation from cancellation
                await asyncio.shield(self.entity_client_service.stop_entity_stream())
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        if self.entity_client_service:
            self.logger.warning("Object deleted before cleanup() was called")

    def compose(self):
        width = 854
        height = 480
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2

        self.condition = AsynchronousCondition(self, name="response_available_condition")
        self.request_queue = AsyncIoQueue(self, name="request_queue")
        self.response_queue = AsynchronousConditionQueue(
            self, name="response_queue", condition=self.condition
        )

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.datapath,
            allocator=RMMAllocator(self, name="video_replayer_allocator"),
            **self.kwargs("replayer"),
        )

        outgoing_requests = GrpcClientRequestOp(
            self,
            name="outgoing_requests",
            request_queue=self.request_queue,
            allocator=RMMAllocator(self, name="pool", device_memory_max_size="256MB"),
        )

        visualizer_op = HolovizOp(
            self,
            name="visualizer_op",
            width=width,
            height=height,
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz"),
        )

        incoming_responses = GrpcClientResponseOp(
            self,
            name="incoming_responses",
            condition=self.condition,
            response_queue=self.response_queue,
        )

        self.add_flow(replayer, outgoing_requests, {("output", "input")})
        self.add_flow(replayer, visualizer_op, {("output", "receivers")})
        self.add_flow(incoming_responses, visualizer_op, {("output", "receivers")})

        self.entity_client_service = EntityClientService(
            str(self.from_config("grpc_client.server_address")),
            str(self.from_config("grpc_client.interrupt")),
            self.request_queue,
            self.response_queue,
            replayer,
        )

    async def start_streaming_client(self):
        while self.entity_client_service is None:
            await asyncio.sleep(0.1)

        await self.entity_client_service.start_entity_stream()
