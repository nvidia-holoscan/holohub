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

from holohub.advanced_network_tx import AdvNetworkOpTx
from holohub.advanced_network_common import (
    adv_net_tx_burst_available,
    adv_net_create_shared_burst_params,
    adv_net_set_hdr,
    adv_net_get_tx_pkt_burst

)

# import holohub.advanced_network_common.holohub as hh
# print(dir(hh))
logger = logging.getLogger("AdvancedNetworkingBench")
logging.basicConfig(level=logging.INFO)

# class AdvNetworkingBenchTxOp : public Operator {
#  public:
#   HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchTxOp)

#   AdvNetworkingBenchTxOp() = default;

#   void initialize() override {
#     HOLOSCAN_LOG_INFO("AdvNetworkingBenchTxOp::initialize()");
#     holoscan::Operator::initialize();

#     size_t buf_size = batch_size_.get() * payload_size_.get();
#     cudaMallocHost(&full_batch_data_h_, buf_size);

#     // Fill in with increasing bytes
#     uint8_t *cptr = static_cast<uint8_t*>(full_batch_data_h_);
#     uint8_t cur = 0;
#     for (int b = 0; b < buf_size; b++) {
#       cptr[b] = cur++;
#     }

#     HOLOSCAN_LOG_INFO("AdvNetworkingBenchTxOp::initialize() complete");
#   }

#   void setup(OperatorSpec& spec) override {
#     spec.output<std::shared_ptr<AdvNetBurstParams>>("burst_out");

#     spec.param<uint32_t>(batch_size_, "batch_size", "Batch size",
#       "Batch size for each processing epoch", 1000);
#     spec.param<uint16_t>(payload_size_, "payload_size", "Payload size",
#       "Payload size to send. Does not include <= L4 headers", 1400);
#   }

#   void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
#     AdvNetStatus ret;

#     /**
#      * Spin waiting until a buffer is free. This can be stalled by sending faster than the NIC can handle it. We
#      * expect the transmit operator to operate much faster than the receiver since it's not having to do any work
#      * to construct packets, and just copying from a buffer into memory.
#     */
#     while (!adv_net_tx_burst_available(batch_size_.get())) {}

#     auto msg = CreateSharedBurstParams();
#     adv_net_set_hdr(msg, port_id, queue_id, batch_size_);

#     if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
#       HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}", static_cast<int>(ret));
#       return;
#     }

#     void *pkt;
#     for (int num_pkt = 0; num_pkt < msg->hdr.num_pkts; num_pkt++) {
#       if ((ret = adv_net_set_cpu_udp_payload( msg,
#                                               num_pkt,
#                                               static_cast<char*>(full_batch_data_h_) +
#                                                     num_pkt * payload_size_.get(),
#                                               payload_size_.get())) != AdvNetStatus::SUCCESS) {
#         HOLOSCAN_LOG_ERROR("Failed to create packet {}", num_pkt);
#       }
#     }

#     op_output.emit(msg, "burst_out");
#   };

#  private:
#   void *full_batch_data_h_;
#   static constexpr uint16_t port_id = 0;
#   static constexpr uint16_t queue_id = 0;
#   Parameter<uint32_t> batch_size_;
#   Parameter<uint16_t> payload_size_;
# };


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
        value = self.index

        while not adv_net_tx_burst_available(self.batch_size):
          continue

        msg = adv_net_create_shared_burst_params()
        adv_net_set_hdr(msg, 0, 0, self.batch_size)

        ret = adv_net_get_tx_pkt_burst(msg)
        if ret != AdvNetStatus.SUCCESS:
          logger.error(f"Error returned from adv_net_get_tx_pkt_burst: {ret}");
          return

# Now define a simple application using the operators defined above
NUM_MSGS = 10


class App(Application):
    def compose(self):
        print("HERE")
        # Define the tx and rx operators, allowing the tx operator to execute 10 times
        if "cfg" in self.kwargs("advanced_network") and "tx" in self.kwargs("advanced_network")["cfg"]:
            tx = AdvancedNetworkingBenchTxOp(self, CountCondition(self, NUM_MSGS), name="tx", **self.kwargs("bench_tx"))
            adv_net_tx = AdvNetworkOpTx(self, name="adv_net_tx")
            self.add_flow(tx, adv_net_tx, {("msg_out", "burst_in")})
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