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

class BasicNetworkOpRx(Operator):
    sock_fd: socket.socket = None
    l4_proto: L4Proto = None
    ip_addr: str = None
    dst_port: int = None
    batch_size: int = None
    max_payload_size: int = None
    connected: bool = False
    send_burst: NetworkOpBurstParams = NetworkOpBurstParams()

    def initialize(self):
        Operator.initialize(self)
        try:
            if (self.l4_proto == "udp"):
                self.l4_proto = L4Proto.UDP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            else:
                self.l4_proto = L4Proto.TCP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            self.sock_fd.bind((self.ip_addr, self.dst_port))
            if self.l4_proto == L4Proto.TCP:
                self.sock_fd.listen(1)
        except socket.error:
            print("FAIL")
            self.logger.error("Failed to create socket")
        print("INIT SUCCESS")
        self.logger.info("Basic RX operator initialized")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("BasicNetworkOpRx")
        logging.basicConfig(level=logging.INFO)

    def setup(self, spec: OperatorSpec):
        spec.param("ip_addr")
        spec.param("dst_port")
        spec.param("l4_proto")
        spec.param("batch_size")
        spec.param("max_payload_size")
        spec.output("burst_out")

    def compute(self, op_input, op_output, context):
        #burst.num_pkts = 1

        if self.l4_proto == L4Proto.TCP and not self.connected:
            try:
                self.sock_fd.settimeout(0.1)
                print("dddd")
                self.conn, self.addr = self.sock_fd.accept()
                
                with self.conn:
                    logger.info(f'Connected by {self.addr}')
                    self.connected = True
            except socket.error:
                return
            finally:
                self.sock_fd.settimeout(None)
        print("CONNECTED")
        while self.send_burst.num_pkts < self.batch_size:
            print("here")
            n = self.conn.recv(self.max_payload_size, flags=socket.MSG_DONTWAIT)
            print("here")
            if len(n) > 0:
                self.send_burst.data.append(n)
                self.send_burst.len      += len(n)
                self.send_burst.num_pkts += 1
            else:
                return

        op_output.emit(self.send_burst, "burst_out")



