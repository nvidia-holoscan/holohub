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
    spec.param<uint32_t>(message_size_,
                         "message_size",
                         "Message size",
                         "Message size in bytes",
                         1024);
    spec.param<std::string>(server_addr_str_,
                         "server_address",
                         "Server address",
                         "Server address",
                         std::string("192.168.3.1"));
    spec.param<uint16_t>(server_port_,
                         "server_port",
                         "Server port",
                         "Server port",
                         4096);
    spec.param<bool>(writes_,
                     "writes",
                     "Writes",
                     "Writes",
                     true);
    spec.param<bool>(reads_,
                     "reads",
                     "Reads",
                     "Reads",
                     false);
  }


  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    Status ret;

    if (conn_id_ == 0) {
      auto ret = rdma_get_server_conn_id(server_addr_str_, server_port_, 0, &conn_id_);
      if (ret != Status::SUCCESS) {
        HOLOSCAN_LOG_INFO("Server connection ID not ready");
        sleep(1);
        return;
      }
      else {
        HOLOSCAN_LOG_INFO("Server connection ID: {}", (void*)conn_id_);
      }
    }
    else if (outstanding_completions < 5) {
      // Post some receive buffers
      auto msg = create_burst_params();

      ret = rdma_set_header(msg, RDMAOpCode::RECEIVE, conn_id_, true, 1, wr_id++, "DATA_RX_CPU_SERVER");

      outstanding_completions++;

      while ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {}

      set_packet_lengths(msg, 0, {message_size_.get()});

      HOLOSCAN_LOG_DEBUG("Posting receive with client cmid {}", (void*)conn_id_);      
      outstanding_tx_wr_ids_[msg->rdma_hdr.wr_id] = msg;      
      send_tx_burst(msg);  
    }

    BurstParams* burst;
    if (get_rx_burst(&burst, conn_id_, true) == Status::SUCCESS) {
      outstanding_completions--;

      uint64_t received_wr_id = burst->rdma_hdr.wr_id;
      HOLOSCAN_LOG_DEBUG("Received completion for WR ID: {}", received_wr_id);

      // Find the received WR ID in the list of outstanding IDs
      auto it = outstanding_tx_wr_ids_.find(received_wr_id);
      if (it != outstanding_tx_wr_ids_.end()) {
        // Found the ID, remove it from the vector
        HOLOSCAN_LOG_DEBUG("Found and removing matching outstanding WR ID: {} with pointer {}", received_wr_id, (void*)it->second->pkts[0][0]);

        free_tx_burst(it->second);
        outstanding_tx_wr_ids_.erase(it);
      } else {
        // This might happen if the completion arrived unexpectedly or was already processed
        HOLOSCAN_LOG_WARN("Received completion for WR ID {}, but it was not found in the outstanding list.", received_wr_id);
      }

      free_rx_burst(burst);
    }
  }

 private:
  std::unordered_map<uint64_t, BurstParams*> outstanding_tx_wr_ids_;
  uint64_t wr_id = 0x2345;
  int outstanding_completions = 0;
  uintptr_t conn_id_ = 0;
  uint16_t port_id_ = 0;
  uint16_t queue_id_ = 0;
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  uint32_t server_addr_;
  Parameter<uint32_t> message_size_;               // Message size in bytes
  Parameter<std::string> server_addr_str_;         // Server address
  Parameter<uint16_t> server_port_;               // Server port
  Parameter<bool> writes_;                        // Writes
  Parameter<bool> reads_;                         // Reads
};

}  // namespace holoscan::ops