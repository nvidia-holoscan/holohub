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
from queue import Queue
from threading import Thread
from typing import Optional

import grpc
from holoscan.conditions import BooleanCondition
from holoscan.core import Operator

from holohub.grpc_operators import holoscan_pb2, holoscan_pb2_grpc


class EntityClientService:
    def __init__(
        self,
        server_address: str,
        interrupt: bool,
        request_queue: Queue,
        response_queue: Queue,
        source_operator: Operator,
    ):
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing Entity Client Service - address: {server_address}")
        self.server_address: str = server_address
        self.interrupt: bool = interrupt
        self.request_queue: Queue = request_queue
        self.response_queue: Queue = response_queue
        self.source_operator: Operator = source_operator
        self.streaming_thread: Optional[Thread] = None
        self.streaming_status_thread: Optional[Thread] = None
        self.timeout_seconds = 5000
        self.abort_connection = False

    async def start_entity_stream(self):
        try:
            self.logger.debug("grpc: Starting streaming client")

            async with grpc.aio.insecure_channel(self.server_address) as self.channel:
                await self.check_channel_availability()
                stub = holoscan_pb2_grpc.EntityStub(self.channel)
                stream: grpc.aio.StreamStreamCall = stub.EntityStream()
                await self._check_connection_status(stream)

                read = asyncio.create_task(self._read_from_stream(self.channel, stream))

                try:
                    write = asyncio.create_task(self._write_to_stream(self.channel, stream))
                    await write
                except Exception as e:
                    read.cancel()
                    write.cancel()
                    raise e
                await read
        except Exception as e:
            self.logger.error(f"grpc: please exit the application with CTRL+C: {e}")
            await self.channel.close()
            asyncio.get_running_loop().stop()
            # call operator.executor().interrupt() to stop the application once the API is available

    async def _check_connection_status(self, stream: grpc.aio.StreamStreamCall):
        try:
            self.logger.debug("grpc: Checking connection status")
            await stream.wait_for_connection()
        except grpc.aio.AioRpcError as e:
            raise grpc.aio.AioRpcError(
                f"{type(self).__name__}: Error waiting for connection to {self.server_address}: {e}"
            ) from e

    async def check_channel_availability(self):
        try:
            self.logger.debug("grpc: Checking channel availability")
            await asyncio.wait_for(self.channel.channel_ready(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(
                f"{type(self).__name__}: Timeout waiting for {self.timeout_seconds} seconds for channel to be ready."
            ) from e

    async def _read_from_stream(self, channel: grpc.aio.Channel, stream: grpc.aio.StreamStreamCall):
        max_retries = 3
        retry = 0
        while True:
            try:
                self.logger.debug("grpc:_read_from_stream: Reading from stream")
                response = await stream.read()
                if response == grpc.aio.EOF:
                    self.logger.debug("grpc:_read_from_stream: End of stream reached")
                    break
                self.response_queue.push(response)
                if response.end_of_stream:
                    self.logger.debug("grpc:_read_from_stream: End of stream reached")
                    break
            except grpc.aio.AioRpcError as e:
                self.logger.debug(f"grpc: server error: {e}")
                retry += 1
                if retry > max_retries:
                    self.logger.error("grpc: Max retries reached. Stopping streaming.")
                    self.abort_connection = True
                    break
            except Exception as e:
                self.logger.error(f"grpc:_read_from_stream: Error reading from stream: {e}")
                break

    async def _write_to_stream(self, channel: grpc.aio.Channel, stream: grpc.aio.StreamStreamCall):
        while True:
            try:
                if self.abort_connection:
                    raise RuntimeError("grpc: aborting connection due to error")

                self.logger.debug("grpc:_write_to_stream: ready to write to stream")
                if not self.request_queue.empty():
                    request = await self.request_queue.pop()
                    self.logger.debug("grpc:_write_to_stream: writing to stream")
                    await stream.write(request)

                    if request.end_of_stream:
                        self.logger.debug("grpc:_write_to_stream: End of stream reached")
                        break
                elif self._end_of_video_reached():
                    self.logger.debug("grpc: sending end_of_stream request")
                    await self.request_queue.push(holoscan_pb2.EntityRequest(end_of_stream=True))
                else:
                    await asyncio.sleep(0.001)
            except grpc.aio.AioRpcError as e:
                self.logger.error(f"grpc: Error writing to stream: {e}")
                break
            except Exception as e:
                raise e

    def _end_of_video_reached(self) -> bool:
        boolean_scheduling_term = next(
            (
                condition
                for _, condition in self.source_operator.conditions.items()
                if type(condition) is BooleanCondition
            ),
            None,
        )

        return (
            not boolean_scheduling_term.check_tick_enabled() if boolean_scheduling_term else False
        )

    async def stop_entity_stream(self):
        try:
            await self.channel.close()
        except Exception as e:
            self.logger.error(f"grpc: Failed to close gRPC channel {e}")
