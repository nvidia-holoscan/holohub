# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
from time import sleep

from holoscan.core import Operator, OperatorSpec

from .basic_network_common import L4Proto


class BasicNetworkOpTx(Operator):
    ip_addr_bytes: bytes = None
    sock_fd: socket.socket = None
    l4_proto: L4Proto = None
    ip_addr: str = None
    dst_port: int = None
    min_ipg_ns: int = None
    max_payload_size: int = None
    retry_connect: int = None
    connected: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("BasicNetworkOpTx")
        self._initialized = False
        logging.basicConfig(level=logging.INFO)

    def _initialize(self):
        try:
            if self.l4_proto == "udp":
                self.l4_proto = L4Proto.UDP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            else:
                self.l4_proto = L4Proto.TCP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        except socket.error:
            self.logger.error("Failed to create socket")

        self.logger.info("Basic TX operator initialized")
        self._initialized = True

    def setup(self, spec: OperatorSpec):
        spec.param("ip_addr")
        spec.param("dst_port")
        spec.param("l4_proto")
        spec.param("max_payload_size")
        spec.param("min_ipg_ns")
        spec.param("retry_connect")
        spec.input("burst_in")

    def compute(self, op_input, op_output, context):
        if not self._initialized:
            self._initialize()

        burst: bytearray = op_input.receive("burst_in")
        if self.l4_proto == L4Proto.TCP:
            if not self.connected:
                try:
                    self.sock_fd.connect((self.ip_addr, self.dst_port))
                    self.connected = True
                    self.logger.info(
                        f"Successfully connected to server at " f"{self.ip_addr}:{self.dst_port}"
                    )
                except socket.error:
                    self.logger.warn(
                        f"Failed to connect to TCP server at "
                        f"{self.ip_addr}:{self.dst_port}. Retrying...."
                    )
                    sleep(self.retry_connect)
                    return

            ttl = 0
            while ttl < len(burst):
                sent = self.sock_fd.send(burst[ttl:])
                if sent == 0:
                    raise RuntimeError("socket connection broken")
                ttl = ttl + sent

        else:
            ttl = 0
            while ttl < len(burst):
                sent = self.sock_fd.sendto(burst[ttl:], (self.ip_addr, self.dst_port))
                if sent == 0:
                    raise RuntimeError("socket connection broken")
                ttl = ttl + sent
