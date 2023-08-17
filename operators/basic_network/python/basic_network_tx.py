# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Operator, OperatorSpec
from .basic_network_common import NetworkOpBurstParams, L4Proto
import socket
import logging
from time import sleep

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
        logging.basicConfig(level=logging.INFO)

    def initialize(self):
        Operator.initialize(self)
        try:
            if (self.l4_proto == "udp"):
                self.l4_proto = L4Proto.UDP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            else:
                self.l4_proto = L4Proto.TCP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        except socket.error:
            self.logger.error("Failed to create socket")

        self.logger.info("Basic TX operator initialized")

    def setup(self, spec: OperatorSpec):
        spec.param("ip_addr")
        spec.param("dst_port")
        spec.param("l4_proto")
        spec.param("max_payload_size")
        spec.param("min_ipg_ns")
        spec.param("retry_connect")
        spec.input("burst_in")

    def compute(self, op_input, op_output, context):
        burst: NetworkOpBurstParams = op_input.receive("burst_in")
        print("HERE")
        if self.l4_proto == L4Proto.TCP:
            while not self.connected:
                try:
                    self.sock_fd.create_connection((self.ip_addr, self.dst_port), timeout=self.retry_connect)
                    self.connected = True
                    self.logger.info(f"Successfully connected to server at {self.ip_addr}:{self.port}")
                except:
                    self.logger.warn(f'Failed to connect to TCP server. Retrying....')
                    sleep(self.retry_connect)
                    return


        sent = self.sock_fd.sendall(burst.data)
        if sent == None:
            self.logger.error("Failed to send data")

