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
    HOLOSCAN_LOG_INFO(
        "Finished receiver with {}/{} bytes/packets received and {}/{} bytes/packets sent",
        ttl_bytes_recv_,
        ttl_pkts_recv_,
        ttl_bytes_sent_,
        ttl_pkts_sent_);

    HOLOSCAN_LOG_INFO("ANO benchmark client op shutting down");
    freeResources();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::initialize()");
    holoscan::Operator::initialize();

    if (send_.get()) {
      send_mr_name_ = server_.get() ? "DATA_TX_GPU_SERVER" : "DATA_TX_GPU_CLIENT";
    }
    if (receive_.get()) {
      receive_mr_name_ = server_.get() ? "DATA_RX_GPU_SERVER" : "DATA_RX_GPU_CLIENT";
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::initialize() complete in {} mode",
                      server_.get() ? "server" : "client");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::freeResources() start");
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaOp::freeResources() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.param<int>(message_size_, "message_size", "Message size", "Message size in bytes", 1024);
    spec.param<std::string>(
        server_addr_str_, "server_address", "Server IP address", "Server IP address", "192.168.3.1");
    spec.param<std::string>(
        client_addr_str_, "client_address", "Client IP address", "Client IP address", "192.168.2.1");
    spec.param<uint16_t>(server_port_, "server_port", "Server port", "Server port", 4096);
    spec.param<bool>(server_, "server", "Server", "Server", false);
    spec.param<bool>(send_, "send", "Send", "Send", false);
    spec.param<bool>(receive_, "receive", "Receive", "Receive", false);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    BurstParams* burst;

    // Establish connection. If we're a client we connect to the server. If we're a server we ask
    // for a connection ID from the advanced_network library.
    if (conn_id_ == 0) {
      if (!server_.get()) {
        HOLOSCAN_LOG_INFO(
            "Connecting to server at {}:{}", server_addr_str_.get(), server_port_.get());
        auto res = rdma_connect_to_server(
            server_addr_str_.get(), server_port_.get(), client_addr_str_.get(), &conn_id_);
        if (res != Status::SUCCESS) {
          HOLOSCAN_LOG_CRITICAL("Failed to connect to server: {}", (int)res);
          conn_id_ = 0;
          return;
        } else {
          HOLOSCAN_LOG_INFO("Connected to server at {}:{} with ID: {}",
                            server_addr_str_.get(),
                            server_port_.get(),
                            (void*)conn_id_);
        }
      } else {
        auto ret = rdma_get_server_conn_id(server_addr_str_.get(), server_port_.get(), &conn_id_);
        if (ret != Status::SUCCESS) {
          HOLOSCAN_LOG_INFO("Server connection ID not ready");
          sleep(1);
          return;
        } else {
          HOLOSCAN_LOG_INFO("Server connection ID: {}", (void*)conn_id_);
        }
      }
    }

    // SEND and RECEIVE use almost the same code, so we can use a lambda to handle both
    auto process_post_msg =
        [&](int& completion_cnt, uint64_t& wr_id, RDMAOpCode opcode, const std::string& mr_name) {
          if (completion_cnt < MAX_OUTSTANDING_COMPLETIONS) {
            auto msg = create_burst_params();

            Status ret =
                rdma_set_header(msg, opcode, conn_id_, server_.get(), 1, wr_id, mr_name.c_str());

            while ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {}

            // Set the length the same as the buffer size
            set_packet_lengths(msg, 0, {message_size_.get()});
            send_tx_burst(msg);

            completion_cnt++;
            wr_id++;
          }
        };

    if (send_.get()) {
      process_post_msg(outstanding_send_completions, send_wr_id, RDMAOpCode::SEND, send_mr_name_);
    }

    if (receive_.get()) {
      process_post_msg(
          outstanding_receive_completions, receive_wr_id, RDMAOpCode::RECEIVE, receive_mr_name_);
    }

    // Process any completions
    if (get_rx_burst(&burst, conn_id_, server_.get()) == Status::SUCCESS) {
      if (rdma_get_opcode(burst) == RDMAOpCode::RECEIVE) {
        outstanding_receive_completions--;
        ttl_bytes_recv_ += get_packet_length(burst, 0);
        ttl_pkts_recv_++;
      } else if (rdma_get_opcode(burst) == RDMAOpCode::SEND) {
        outstanding_send_completions--;
        ttl_bytes_sent_ += get_packet_length(burst, 0);
        ttl_pkts_sent_++;
      }

      uint64_t received_wr_id = burst->rdma_hdr.wr_id;

      if (burst->rdma_hdr.status != Status::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Received completion for WR ID: {} with status: {}",
                           received_wr_id,
                           (int)burst->rdma_hdr.status);
      }

      free_tx_burst(burst);
    }
  }

 private:
  static constexpr int MAX_OUTSTANDING_COMPLETIONS = 5;
  std::string send_mr_name_ = "";
  std::string receive_mr_name_ = "";
  int outstanding_send_completions = 0;
  int outstanding_receive_completions = 0;
  uint64_t send_wr_id = 0x1234;
  uint64_t receive_wr_id = 0x2345;
  int64_t ttl_bytes_recv_ = 0;  // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;   // Total packets received in operator
  int64_t ttl_bytes_sent_ = 0;  // Total bytes sent in operator
  int64_t ttl_pkts_sent_ = 0;   // Total packets sent in operator
  uintptr_t conn_id_ = 0;
  Parameter<bool> server_;
  Parameter<int> message_size_;             // Message size in bytes
  Parameter<std::string> server_addr_str_;  // Server address
  Parameter<std::string> client_addr_str_;  // Client address
  Parameter<uint16_t> server_port_;         // Server port
  Parameter<bool> send_;
  Parameter<bool> receive_;
};

}  // namespace holoscan::ops
