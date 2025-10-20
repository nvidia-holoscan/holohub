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
from concurrent import futures
from queue import Queue
from typing import List, Optional

import grpc
from grpc_health.v1 import health, health_pb2_grpc

import holohub.grpc_operators.holoscan_pb2_grpc
from operators.grpc_operators.python.server.application_factory import ApplicationFactory
from operators.grpc_operators.python.server.entity_servicer import HoloscanEntityServicer
from operators.grpc_operators.python.server.grpc_application import HoloscanGrpcApplication


class GrpcService:
    def __new__(cls):
        """
        Create a singleton instance of the GrpcService class
        """
        if not hasattr(cls, "instance"):
            cls.instance = super(GrpcService, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.server: Optional[grpc.aio.Server] = None
        self.services: Optional[List[HoloscanEntityServicer]] = None
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.__initialized: bool = True

    def initialize(self, port: int, application_factory: ApplicationFactory):
        self.server_address: str = f"0.0.0.0:{port}"
        self.application_factory: ApplicationFactory = application_factory

    async def start(
        self, services: List[HoloscanEntityServicer], enable_health_check_service: bool = True
    ):
        if len(services) == 0:
            raise ValueError("At least one service must be provided")

        self.services = services
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        for service in self.services:
            service.configure_callbacks(
                self._create_application_instance, self._destroy_application_instance
            )
            holohub.grpc_operators.holoscan_pb2_grpc.add_EntityServicer_to_server(
                service, self.server
            )
        if enable_health_check_service:
            health_pb2_grpc.add_HealthServicer_to_server(health.HealthServicer(), self.server)

        self.server.add_insecure_port(self.server_address)
        await self.server.start()
        self.logger.info(f"grpc: Server listening on {self.server_address}")
        await self.server.wait_for_termination()

    async def stop(self):
        self.logger.info("grpc: Server shutting down")
        await self.server.stop(None)

    def _create_application_instance(
        self, service_name: str, incoming_request_queue: Queue, outgoing_response_queue: Queue
    ):
        return self.application_factory.create_new_application_instance(
            service_name, incoming_request_queue, outgoing_response_queue
        )

    def _destroy_application_instance(self, application_instance: HoloscanGrpcApplication):
        self.application_factory.destroy_application_instance(application_instance)
