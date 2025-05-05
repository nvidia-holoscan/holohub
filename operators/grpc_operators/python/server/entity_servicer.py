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
from typing import AsyncIterable, Callable, Optional

import grpc

from holohub.grpc_operators.holoscan_pb2 import EntityRequest, EntityResponse
from holohub.grpc_operators.holoscan_pb2_grpc import EntityServicer
from operators.grpc_operators.python.server.grpc_application import HoloscanGrpcApplication


class StreamingStatus:
    def __init__(self):
        self.last_received_frame_no = -1
        self.outgoing_frames = 0


class HoloscanEntityServicer(EntityServicer):
    def __init__(
        self,
        application_name: str,
    ):
        self.application_name: str = application_name
        self.new_entity_stream_rpc: Optional[
            Callable[[str, Queue, Queue], HoloscanGrpcApplication]
        ] = None
        self.entity_stream_rpc_complete: Optional[Callable[[HoloscanGrpcApplication], None]] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

    def configure_callbacks(
        self,
        new_entity_stream_rpc: Callable[[str, Queue, Queue], HoloscanGrpcApplication],
        entity_stream_rpc_complete: Callable[[HoloscanGrpcApplication], None],
    ):
        self.new_entity_stream_rpc = new_entity_stream_rpc
        self.entity_stream_rpc_complete = entity_stream_rpc_complete

    async def EntityStream(
        self, request_iterator: AsyncIterable[EntityRequest], context: grpc.aio.ServicerContext
    ) -> AsyncIterable[EntityResponse]:
        incoming_request_queue: Queue = Queue()
        outgoing_response_queue: Queue = Queue()
        app = self.new_entity_stream_rpc(
            self.application_name, incoming_request_queue, outgoing_response_queue
        )

        if app is None:
            raise RuntimeError("Failed to create application instance")

        while not app.composed:
            self.logger.debug("Waiting for application to be composed")
            await asyncio.sleep(0.5)

        status = StreamingStatus()
        request_awaiter = asyncio.create_task(
            self._process_requests(app, request_iterator, context, status)
        )
        response_awaiter = asyncio.create_task(self._process_responses(app, context, status))

        try:
            await asyncio.gather(request_awaiter, response_awaiter)
        except asyncio.CancelledError:
            self.logger.warning("grpc: EntityStream - client cancelled")
        except Exception as ex:
            self.logger.error(f"grpc: EntityStream - exception occurred: {ex}")
            raise ex
        finally:
            self.entity_stream_rpc_complete(app)

    async def _process_responses(
        self,
        app: HoloscanGrpcApplication,
        context: grpc.aio.ServicerContext,
        status: StreamingStatus,
    ):
        while True:
            if app.is_response_available():
                response: EntityResponse = app.dequeue_response()
                await context.write(response)
                status.outgoing_frames += 1
                self.logger.debug(
                    f"grpc: outgoing_frame {status.outgoing_frames} - last received frame {status.last_received_frame_no}"
                )
                if response.end_of_stream:
                    break
            else:
                await asyncio.sleep(0.01)

    async def _process_requests(self, app, request_iterator, context, status):
        try:
            async for request in request_iterator:
                if request.end_of_stream:
                    self.logger.debug("grpc: EntityStream - end of stream received")
                    await asyncio.to_thread(
                        app.enqueue_response, EntityResponse(end_of_stream=True)
                    )
                    break
                else:
                    self.logger.debug("grpc: EntityStream - new request received")
                    status.last_received_frame_no = request.frame_no
                    await asyncio.to_thread(app.enqueue_request, request)
                    self.logger.debug(
                        f"grpc: outgoing_frame {status.outgoing_frames} - last received frame {status.last_received_frame_no}"
                    )
        except Exception as ex:
            self.logger.error(f"grpc: EntityStream - exception occurred: {ex}")
