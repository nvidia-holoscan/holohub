# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ipaddress
import logging
from concurrent import futures
from queue import Queue

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
            cls.instance = super().__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        if self.__initialized:
            return
        self.server: grpc.aio.Server | None = None
        self.services: list[HoloscanEntityServicer] | None = None
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.__initialized: bool = True

    def initialize(
        self,
        port: int,
        application_factory: ApplicationFactory,
        *,
        host: str = "127.0.0.1",
        private_key: bytes | None = None,
        certificate_chain: bytes | None = None,
        root_certificates: bytes | None = None,
    ):
        """Allow plaintext on loopback only; remote listeners require mutual TLS.

        TLS inputs are PEM bytes. ``root_certificates`` must contain the CA
        certificates trusted to issue client certificates.
        """
        try:
            # Pin this alias to loopback; never resolve hostnames for a listener.
            address = ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)
        except ValueError as exc:
            raise ValueError("The gRPC bind host must be an IP address or localhost") from exc
        port = int(port)
        if not 0 <= port <= 65535:
            raise ValueError("The gRPC port must be between 0 and 65535")

        tls = (private_key, certificate_chain, root_certificates)
        credentials = None
        if any(value is not None for value in tls):
            if not all(tls):
                raise ValueError(
                    "Mutual TLS requires a private key, certificate chain, and client CA"
                )
            credentials = grpc.ssl_server_credentials(
                [(private_key, certificate_chain)],
                root_certificates=root_certificates,
                require_client_auth=True,
            )
        elif not address.is_loopback:
            raise ValueError("A non-loopback gRPC listener requires mutual TLS credentials")

        self.server_address: str = (
            f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
        )
        self.credentials = credentials
        self.application_factory: ApplicationFactory = application_factory

    async def start(
        self, services: list[HoloscanEntityServicer], enable_health_check_service: bool = True
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

        if self.credentials is None:
            bound_port = self.server.add_insecure_port(self.server_address)
        else:
            bound_port = self.server.add_secure_port(self.server_address, self.credentials)
        if not bound_port:
            raise RuntimeError(f"Failed to bind gRPC server to {self.server_address}")
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
