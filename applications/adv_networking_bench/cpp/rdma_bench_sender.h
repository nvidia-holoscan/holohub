/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "adv_network_rx.h"
#include "adv_network_kernels.h"
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>


namespace holoscan::ops {

class AdvNetworkingRdmaServerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingRdmaServerOp)

  AdvNetworkingRdmaServerOp() = default;

  ~AdvNetworkingRdmaServerOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received",
                      ttl_bytes_recv_,
                      ttl_pkts_recv_);

    HOLOSCAN_LOG_INFO("ANO benchmark server op shutting down");
    freeResources();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaServerOp::initialize()");
    holoscan::Operator::initialize();

    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaServerOp::initialize() complete");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaServerOp::freeResources() start");
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaServerOp::freeResources() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.input<AdvNetBurstParams*>("rdma_in");
    spec.output<AdvNetBurstParams*>("rdma_out");
    spec.param<uint32_t>(batch_size_,
                         "batch_size",
                         "Batch size",
                         "Batch size in packets for each processing epoch",
                         1000);
    spec.param<uint16_t>(max_packet_size_,
                         "max_packet_size",
                         "Max packet size",
                         "Maximum packet size expected from sender",
                         9100);
  }


  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    // Get new input burst (ANO batch of packets)
    auto burst_opt = op_input.receive<AdvNetBurstParams*>("burst_in");
    if (!burst_opt) { return; }

    auto burst = burst_opt.value();
    auto burst_size = adv_net_get_num_pkts(burst);

    // Count packets received
    ttl_pkts_recv_ += burst_size;
    ttl_bytes_recv_ += adv_net_get_pkt_len(burst, 0);    

    printf("Got %d bytes\n", adv_net_get_pkt_len(burst, 0));
  }

 private:
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  Parameter<uint32_t> batch_size_;                 // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;            // Maximum size of a single packet
};

}  // namespace holoscan::ops
