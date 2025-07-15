# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import cupy
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from holohub.advanced_network_common import (
    Status,
    create_burst_params,
    get_num_packets,
    get_tx_packet_burst,
    is_tx_burst_available,
    set_cpu_udp_payload,
    set_header,
)
from holohub.advanced_network_tx import AdvNetworkOpTx

logger = logging.getLogger("AdvancedNetworkingBench")
logging.basicConfig(level=logging.INFO)


class AdvancedNetworkingBenchTxOp(Operator):
    def __init__(self, fragment, *args, batch_size, payload_size, **kwargs):
        self.index = 1
        self.batch_size = batch_size
        self.payload_size = payload_size
        self.buf_size = self.batch_size * self.payload_size
        self.buf = cupy.cuda.alloc_pinned_memory(self.buf_size)
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        Operator.initialize(self)

    def setup(self, spec: OperatorSpec):
        spec.output("msg_out")

    def compute(self, op_input, op_output, context):
        while not is_tx_burst_available(self.batch_size):
            continue

        msg = create_burst_params()
        set_header(msg, 0, 0, self.batch_size)

        ret = get_tx_packet_burst(msg)
        if ret != Status.SUCCESS:
            logger.error(f"Error returned from get_tx_packet_burst: {ret}")
            return

        for num_pkt in range(get_num_packets(msg)):
            ret = set_cpu_udp_payload(
                msg, num_pkt, self.buf.ptr + num_pkt * self.payload_size, self.payload_size
            )
            if ret != Status.SUCCESS:
                logger.error(
                    f"Error returned from set_cpu_udp_payload: " f"{ret} != {Status.SUCCESS}"
                )
                return
        print(type(msg))
        op_output.emit(msg, "msg_out")


# Now define a simple application using the operators defined above
NUM_MSGS = 10


class App(Application):
    def compose(self):
        print("HERE")
        # Define the tx and rx operators, allowing the tx operator to execute 10 times
        if (
            "cfg" in self.kwargs("advanced_network")
            and "tx" in self.kwargs("advanced_network")["cfg"]
        ):
            bench_tx = AdvancedNetworkingBenchTxOp(
                self, CountCondition(self, NUM_MSGS), name="bench_tx", **self.kwargs("bench_tx")
            )
            adv_net_tx = AdvNetworkOpTx(self, name="adv_net_tx")
            self.add_flow(bench_tx, adv_net_tx, {("msg_out", "burst_in")})
        else:
            logger.info("No TX config found")

        # if len(self.kwargs("network_rx")) > 0:
        #     basic_net_rx = BasicNetworkOpRx(self, name="basic_net_rx", **self.kwargs("network_rx"))
        #     rx = BasicNetworkPingRxOp(self, name="rx")
        #     self.add_flow(basic_net_rx, rx, {("burst_out", "msg_in")})
        # else:
        #     logger.info("No RX config found")


if __name__ == "__main__":
    config_path = sys.argv[1]
    app = App()
    app.config(config_path)
    app.run()
