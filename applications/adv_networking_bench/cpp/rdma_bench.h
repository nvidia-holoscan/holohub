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

class AdvNetworkingRdmaOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingRdmaOp)

  AdvNetworkingRdmaOp() = default;

  ~AdvNetworkingRdmaOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received",
                      ttl_bytes_recv_,
                      ttl_pkts_recv_);

    HOLOSCAN_LOG_INFO("ANO benchmark clent op shutting down");
    freeResources();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::initialize()");
    holoscan::Operator::initialize();

    if (send_.get()) {
      send_mr_name_ = server_.get() ? "DATA_TX_CPU_SERVER" : "DATA_TX_CPU_CLIENT";
    }
    if (receive_.get()) {  
      receive_mr_name_ = server_.get() ? "DATA_RX_CPU_SERVER" : "DATA_RX_CPU_CLIENT";
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::initialize() complete in {} mode", 
      server_.get() ? "server" : "client");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::freeResources() start");
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::freeResources() complete");
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
                            "192.168.3.1");
    spec.param<std::string>(client_addr_str_,
                            "client_address",
                            "Client address",
                            "Client address",
                            "192.168.2.1");                       
    spec.param<uint16_t>(server_port_,
                         "server_port",
                         "Server port",
                         "Server port",
                         4096);
    spec.param<bool>(server_,
                     "server",
                     "Server",
                     "Server",
                     false);
    spec.param<bool>(send_,
                     "send",
                     "Send",
                     "Send",
                     false);
    spec.param<bool>(receive_,
                     "receive",
                     "Receive",
                     "Receive",
                     false);
  }


  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    BurstParams *burst;

    // Establish connection. If we're a client we connect to the server. If we're a server we ask
    // for a connection ID from the ANO.
    if (conn_id_ == 0) {
      if (!server_.get()) {
        HOLOSCAN_LOG_INFO("Connecting to server at {}:{}", server_addr_str_.get(), server_port_.get());
        auto res = rdma_connect_to_server(server_addr_str_.get(), server_port_.get(), client_addr_str_.get(), &conn_id_);
        if (res != Status::SUCCESS) {
          HOLOSCAN_LOG_CRITICAL("Failed to connect to server: {}", (int)res);
          conn_id_ = 0;
          return;
        }
        else {
          HOLOSCAN_LOG_INFO("Connected to server at {}:{} with ID: {}", server_addr_str_.get(), server_port_.get(), (void*)conn_id_);     
        }
      }
      else {
        auto ret = rdma_get_server_conn_id(server_addr_str_.get(), server_port_.get(), &conn_id_);
        if (ret != Status::SUCCESS) {
          HOLOSCAN_LOG_INFO("Server connection ID not ready");
          sleep(1);
          return;
        }
        else {
          HOLOSCAN_LOG_INFO("Server connection ID: {}", (void*)conn_id_);
        }
      }
    }

    Status ret;

    // Process any SEND types
    if (send_.get()) {
      if (outstanding_send_completions < 5) {
        auto msg = create_burst_params();

        outstanding_send_completions++;
        ret = rdma_set_header(msg, RDMAOpCode::SEND, conn_id_, server_.get(), 1, send_wr_id, send_mr_name_.c_str());

        while ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {}

        // Set the length the same as the buffer size
        set_packet_lengths(msg, 0, {message_size_.get()});
        send_tx_burst(msg);

        outstanding_send_wr_ids_[send_wr_id] = msg;
        send_wr_id++;
      }
    }

    // Process any RECEIVE types
    if (receive_.get()) {
      if (outstanding_receive_completions < 5) {
        auto msg = create_burst_params();

        outstanding_receive_completions++;
        ret = rdma_set_header(msg, RDMAOpCode::RECEIVE, conn_id_, server_.get(), 1, receive_wr_id, receive_mr_name_.c_str());

        while ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {}

        // Set the length the same as the buffer size
        set_packet_lengths(msg, 0, {message_size_.get()});
        send_tx_burst(msg);

        outstanding_receive_wr_ids_[receive_wr_id] = msg;
        receive_wr_id++;  
      }
    }

    // Process any completions
    if (get_rx_burst(&burst, conn_id_, server_.get()) == Status::SUCCESS) {
      if (rdma_get_opcode(burst) == RDMAOpCode::RECEIVE) {
        outstanding_receive_completions--;

        uint64_t received_wr_id = burst->rdma_hdr.wr_id;
        HOLOSCAN_LOG_DEBUG("Received completion for WR ID: {}", received_wr_id);

        // Find the received WR ID in the list of outstanding IDs
        auto it = outstanding_receive_wr_ids_.find(received_wr_id);
        if (it != outstanding_receive_wr_ids_.end()) {
          // Found the ID, remove it from the vector
          HOLOSCAN_LOG_DEBUG("Found and removing matching outstanding WR ID: {}", received_wr_id);
          free_tx_burst(it->second);
          outstanding_receive_wr_ids_.erase(it);
        } else {
          // This might happen if the completion arrived unexpectedly or was already processed
          HOLOSCAN_LOG_WARN("Received completion for WR ID {}, but it was not found in the outstanding list.", received_wr_id);
        }
      }
      else if (rdma_get_opcode(burst) == RDMAOpCode::SEND) {
        outstanding_send_completions--;

        uint64_t send_wr_id = burst->rdma_hdr.wr_id;
        HOLOSCAN_LOG_DEBUG("Received completion for WR ID: {}", send_wr_id);

        // Find the received WR ID in the list of outstanding IDs
        auto it = outstanding_send_wr_ids_.find(send_wr_id);
        if (it != outstanding_send_wr_ids_.end()) {
          // Found the ID, remove it from the vector
          HOLOSCAN_LOG_DEBUG("Found and removing matching outstanding WR ID: {}", send_wr_id);
          free_tx_burst(it->second);
          outstanding_send_wr_ids_.erase(it);
        } else {
          // This might happen if the completion arrived unexpectedly or was already processed
          HOLOSCAN_LOG_WARN("Received completion for WR ID {}, but it was not found in the outstanding list.", send_wr_id);
        }
      }

      free_rx_burst(burst);
    }
  }

 private:
  std::string send_mr_name_ = "";
  std::string receive_mr_name_ = "";
  int outstanding_send_completions = 0;
  int outstanding_receive_completions = 0;
  std::unordered_map<uint64_t, BurstParams*> outstanding_send_wr_ids_;
  std::unordered_map<uint64_t, BurstParams*> outstanding_receive_wr_ids_;
  uint64_t send_wr_id = 0x1234;
  uint64_t receive_wr_id = 0x1234;
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  uintptr_t conn_id_ = 0;
  Parameter<bool> server_;
  Parameter<uint32_t> message_size_;               // Message size in bytes
  Parameter<std::string> server_addr_str_;         // Server address
  Parameter<std::string> client_addr_str_;         // Client address
  Parameter<uint16_t> server_port_;              // Server port
  Parameter<bool> send_;
  Parameter<bool> receive_;
};

}  // namespace holoscan::ops