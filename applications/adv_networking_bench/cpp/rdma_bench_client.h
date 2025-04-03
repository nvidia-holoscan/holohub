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

class AdvNetworkingRdmaClientOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingRdmaClientOp)

  AdvNetworkingRdmaClientOp() = default;

  ~AdvNetworkingRdmaClientOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received",
                      ttl_bytes_recv_,
                      ttl_pkts_recv_);

    HOLOSCAN_LOG_INFO("ANO benchmark clent op shutting down");
    freeResources();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::initialize()");
    holoscan::Operator::initialize();

    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::initialize() complete");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::freeResources() start");
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::freeResources() complete");
  }

  bool connect_to_server(OutputContext& op_output) {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::connect_to_server() start");

    auto burst = adv_net_create_burst();
    auto res = adv_net_rdma_connect_to_server(server_address_str_.get(), server_port_.get(), &conn_id_);
    if (res != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to connect to server: {}", res);
      return false;
    }
    else {
      HOLOSCAN_LOG_INFO("Connected to server {}:{} with ID: {}", server_address_str_.get(), server_port_.get(), conn_id_);
    }


    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::connect_to_server() complete");
    return true;
  }

  void setup(OperatorSpec& spec) override {
    spec.input<AdvNetBurstParams*>("rdma_in");
    spec.output<AdvNetBurstParams*>("rdma_out");
    spec.param<uint32_t>(message_size_,
                         "message_size",
                         "Message size",
                         "Message size in bytes",
                         1024);
    spec.param<bool>(rdma_write_,
                     "rdma_write",
                     "Rdma write",
                     "Whether to issue RDMA writes",
                     true);
    spec.param<std::string>(server_address_str_,
                            "server_address",
                            "Server address",
                            "Server address",
                            "192.168.3.1");
    spec.param<uint16_t>(server_port_,
                         "server_port",
                         "Server port",
                         "Server port",
                         4096);

    server_addr_ = inet_addr(server_address_str_.get().c_str());
  }


  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    // Get new input burst (ANO batch of packets)
    auto burst_opt = op_input.receive<AdvNetBurstParams*>("burst_in");
    if (!burst_opt) { 
      if (!connected_) {
        connect_to_server(op_output);
      }
    }

    auto burst = burst_opt.value();
    auto burst_size = adv_net_get_num_pkts(burst);

    // Count packets received
    ttl_pkts_recv_ += burst_size;
    ttl_bytes_recv_ += adv_net_get_pkt_len(burst, 0);    

    printf("Got %d bytes\n", adv_net_get_pkt_len(burst, 0));
  }

 private:
  bool connected_ = false;
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  uint32_t server_addr_;
  uintptr_t conn_id_ = nullptr;
  Parameter<bool> rdma_write_;               // Message size in bytes
  Parameter<uint32_t> message_size_;               // Message size in bytes
  Parameter<std::string> server_address_str_;         // Server address
  Parameter<uint16_t> server_port_;              // Server port
};

}  // namespace holoscan::ops
