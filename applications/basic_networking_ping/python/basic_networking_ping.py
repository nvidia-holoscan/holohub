# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from holohub.basic_network import BasicNetworkOpRx, BasicNetworkOpTx

logger = logging.getLogger("BasicNetworkingPing")
logging.basicConfig(level=logging.INFO)


class BasicNetworkPingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("msg_out")

    def compute(self, op_input, op_output, context):
        value = self.index
        to_send = list(range(value, value + 10))
        logger.info(f"Sending index {self.index}: {bytearray(to_send)}")
        self.index += 1
        op_output.emit(bytearray(to_send), "msg_out")


class BasicNetworkPingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("msg_in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("msg_in")
        data = list(value.data)
        logger.info(f"Rx message received (count: {self.count}, size: {len(data)}, value:{data})")
        self.count += 1


# Now define a simple application using the operators defined above
NUM_MSGS = 10


class App(Application):
    def compose(self):
        # Define the tx and rx operators, allowing the tx operator to execute 10 times
        if len(self.kwargs("network_tx")) > 0:
            basic_net_tx = BasicNetworkOpTx(self, name="basic_net_tx", **self.kwargs("network_tx"))
            tx = BasicNetworkPingTxOp(self, CountCondition(self, NUM_MSGS), name="tx")
            self.add_flow(tx, basic_net_tx, {("msg_out", "burst_in")})
        else:
            logger.info("No TX config found")

        if len(self.kwargs("network_rx")) > 0:
            basic_net_rx = BasicNetworkOpRx(self, name="basic_net_rx", **self.kwargs("network_rx"))
            rx = BasicNetworkPingRxOp(self, name="rx")
            self.add_flow(basic_net_rx, rx, {("burst_out", "msg_in")})
        else:
            logger.info("No RX config found")


def main():
    if len(sys.argv) != 2:
        logger.error(
            "Must specify configuration file as second argument. "
            "If using the 'run' script, use --extra_args <config_name>"
        )
        sys.exit(-1)

    config_path = sys.argv[1]
    if not Path(config_path).is_file():
        logger.error(f"Configuration file {config_path} not found")
        sys.exit(-2)

    app = App()
    app.config(config_path)
    app.run()


if __name__ == "__main__":
    main()
