# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..# SPDX-License-Identifier: Apache-2.0
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

import copy
import logging
import socket
import time

from holoscan.core import Operator, OperatorSpec

from .basic_network_common import L4Proto, NetworkOpBurstParams


class BasicNetworkOpRx(Operator):
    sock_fd: socket.socket = None
    l4_proto: L4Proto = None
    ip_addr: str = None
    dst_port: int = None
    batch_size: int = None
    max_payload_size: int = None
    max_burst_interval: int = None
    connected: bool = False
    send_burst: NetworkOpBurstParams = NetworkOpBurstParams()
    burst_start_time: float = None

    def _initialize(self):
        try:
            if self.l4_proto == "udp":
                self.l4_proto = L4Proto.UDP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            else:
                self.l4_proto = L4Proto.TCP
                self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            self.sock_fd.bind((self.ip_addr, self.dst_port))
            if self.l4_proto == L4Proto.TCP:
                self.sock_fd.listen(1)

            self.logger.info(f"Successfully listening on {self.ip_addr}:{self.dst_port}")
        except socket.error as err:
            self.logger.error(f"Failed to create socket: {err}")

        self.logger.info("Basic RX operator initialized")
        self._initialized = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("BasicNetworkOpRx")
        self._initialized = False
        logging.basicConfig(level=logging.INFO)

    def setup(self, spec: OperatorSpec):
        spec.param("ip_addr")
        spec.param("dst_port")
        spec.param("l4_proto")
        spec.param("batch_size")
        spec.param("max_payload_size")
        spec.param("max_burst_interval", None)
        spec.output("burst_out")

    def compute(self, op_input, op_output, context):
        if not self._initialized:
            self._initialize()

        if self.l4_proto == L4Proto.TCP and not self.connected:
            try:
                self.sock_fd.settimeout(1)
                self.conn, self.addr = self.sock_fd.accept()
                self.logger.info(f"Connected by {self.addr}")
                self.connected = True
            except socket.error:
                return
            finally:
                self.sock_fd.settimeout(None)

        # Initialize burst start time if this is the first packet in a new burst
        if len(self.send_burst.data) == 0:
            self.burst_start_time = time.time()

        while len(self.send_burst.data) < self.batch_size:
            # Check if max_burst_interval has been reached
            if self.max_burst_interval and self.max_burst_interval > 0 and self.burst_start_time:
                current_time = time.time()
                elapsed_ms = (current_time - self.burst_start_time) * 1000

                if elapsed_ms >= self.max_burst_interval:
                    self.logger.info(f"Timeout, emitting {len(self.send_burst.data)} bytes (elapsed time: {elapsed_ms:.1f} ms)")
                    break

            try:
                if self.l4_proto == L4Proto.UDP:
                    tmp = self.sock_fd.recvfrom(self.max_payload_size, socket.MSG_DONTWAIT)
                    n = tmp[0]
                else:
                    n = self.conn.recv(self.max_payload_size, socket.MSG_DONTWAIT)
            except BlockingIOError:
                if len(self.send_burst.data) > 0:
                    break
                else:
                    return

            if len(n) > 0:
                self.send_burst.data.extend(n)
            else:
                return

        # Emit data when batch is full, timeout occurred, or no more data available
        if len(self.send_burst.data) > 0:
            tmp = copy.deepcopy(self.send_burst)
            op_output.emit(tmp, "burst_out")
            self.send_burst.reset()
