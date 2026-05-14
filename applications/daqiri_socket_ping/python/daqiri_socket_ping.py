# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import struct
import sys
import time
from pathlib import Path

import daqiri
from holoscan.conditions import BooleanCondition, CountCondition
from holoscan.core import Application, Operator, OperatorSpec

logger = logging.getLogger("DaqiriSocketPing")
logging.basicConfig(level=logging.INFO)

NUM_MSGS = 10
POST_PING_FLUSH_MSGS = 128


class DaqiriSocketPingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        self.conn_id = 0
        self.port = 0
        self.queue = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("server_address", "127.0.0.1")
        spec.param("client_address", "127.0.0.1")
        spec.param("server_port", 5001)

    def compute(self, op_input, op_output, context):
        del op_input, op_output, context
        self._ensure_connected()

        burst = daqiri.create_tx_burst_params()
        daqiri.set_header(burst, self.port, self.queue, 1, 1)

        while daqiri.get_tx_packet_burst(burst) != daqiri.Status.SUCCESS:
            time.sleep(0.01)

        value = self.index if self.index < NUM_MSGS else -1
        payload = struct.pack("i", value)
        status = daqiri.copy_buffer_to_segment_packet(burst, 0, 0, payload)
        if status != daqiri.Status.SUCCESS:
            daqiri.free_all_packets_and_burst_tx(burst)
            raise RuntimeError(f"copy_buffer_to_segment_packet failed: {status}")

        daqiri.set_packet_lengths(burst, 0, [len(payload)])
        burst.rdma_conn_id = self.conn_id

        while daqiri.send_tx_burst(burst) != daqiri.Status.SUCCESS:
            time.sleep(0.01)

        if value >= 0:
            logger.info("Ping message sent with value %d", value)
        self.index += 1

    def _ensure_connected(self):
        while self.conn_id == 0:
            status, conn_id = daqiri.socket_connect_to_server(
                self.server_address, self.server_port, self.client_address
            )
            if status != daqiri.Status.SUCCESS:
                time.sleep(0.1)
                continue

            status, port, queue = daqiri.socket_get_port_queue(conn_id)
            if status == daqiri.Status.SUCCESS:
                self.conn_id = conn_id
                self.port = port
                self.queue = queue
                logger.info("Connected to server at %s:%d", self.server_address, self.server_port)
                return

            time.sleep(0.1)


class DaqiriSocketPingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.conn_id = 0
        self.port = 0
        self.queue = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("server_address", "127.0.0.1")
        spec.param("server_port", 5001)

    def compute(self, op_input, op_output, context):
        del op_input, op_output, context
        if not self._ensure_connected():
            return

        status, burst = daqiri.get_rx_burst_for_connection(self.conn_id, True)
        if status != daqiri.Status.SUCCESS or burst is None:
            time.sleep(0.01)
            return

        for pkt_idx in range(daqiri.get_num_packets(burst)):
            status, payload = daqiri.get_packet_bytes(burst, pkt_idx, 4)
            if status != daqiri.Status.SUCCESS:
                continue

            value = struct.unpack("i", payload)[0]
            if value < 0:
                continue
            logger.info("Ping message received with value %d", value)
            if value == NUM_MSGS - 1:
                self.conditions["is_alive"].disable_tick()

        daqiri.free_all_packets_and_burst_rx(burst)

    def _ensure_connected(self):
        if self.conn_id != 0:
            return True

        status, conn_id = daqiri.socket_get_server_conn_id(self.server_address, self.server_port)
        if status != daqiri.Status.SUCCESS:
            time.sleep(0.1)
            return False

        status, port, queue = daqiri.socket_get_port_queue(conn_id)
        if status != daqiri.Status.SUCCESS:
            time.sleep(0.1)
            return False

        self.conn_id = conn_id
        self.port = port
        self.queue = queue
        logger.info("Accepted client connection on %s:%d", self.server_address, self.server_port)
        return True


class App(Application):
    def compose(self):
        if self.kwargs("ping_tx"):
            tx = DaqiriSocketPingTxOp(
                self,
                CountCondition(self, NUM_MSGS + POST_PING_FLUSH_MSGS),
                name="ping_tx",
                **self.kwargs("ping_tx"),
            )
            self.add_operator(tx)
        else:
            logger.info("No TX config found")

        if self.kwargs("ping_rx"):
            rx = DaqiriSocketPingRxOp(
                self,
                BooleanCondition(self, name="is_alive"),
                name="ping_rx",
                **self.kwargs("ping_rx"),
            )
            self.add_operator(rx)
        else:
            logger.info("No RX config found")


def main():
    if len(sys.argv) != 2:
        logger.error(
            "Must specify configuration file as second argument. "
            "If using the 'run' script, use --run-args <config_name>"
        )
        sys.exit(-1)

    config_path = sys.argv[1]
    if not Path(config_path).is_file():
        logger.error("Configuration file %s not found", config_path)
        sys.exit(-2)

    if daqiri.daqiri_init(config_path) != daqiri.Status.SUCCESS:
        logger.error("Failed to configure DAQIRI")
        sys.exit(-3)

    try:
        app = App()
        app.config(config_path)
        app.run()
        daqiri.print_stats()
    finally:
        daqiri.shutdown()


if __name__ == "__main__":
    main()
