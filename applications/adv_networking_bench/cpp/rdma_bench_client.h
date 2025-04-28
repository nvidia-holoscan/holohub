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

    server_addr_ = inet_addr(server_addr_str_.get().c_str());
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::initialize() complete");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::freeResources() start");
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::freeResources() complete");
  }

  bool connect_to_server(OutputContext& op_output) {
    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::connect_to_server() start");

    auto burst = create_burst_params();
    auto res = rdma_connect_to_server(server_addr_str_.get(), server_port_.get(), &conn_id_);
    if (res != Status::SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to connect to server: {}", (int)res);
      return false;
    }
    else {
      HOLOSCAN_LOG_INFO("Connected to server {}:{} with ID: {}", server_addr_str_.get(), server_port_.get(), conn_id_);
    }


    HOLOSCAN_LOG_INFO("AdvNetworkingRdmaClientOp::connect_to_server() complete");
    return true;
  }

  void setup(OperatorSpec& spec) override {
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


  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    // Get new input burst (ANO batch of packets)
    // auto burst_opt = op_input.receive<BurstParams*>("burst_in");
    // if (!burst_opt) { 
    //   if (!connected_) {
    //     connect_to_server(op_output);
    //   }
    // }

    if (conn_id_ == 0) {
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

    Status ret;
    if (!sent) {
      for (int i = 0; i < 10; i++) {
        auto msg = create_burst_params();

        msg->rdma_hdr.opcode = AdvNetRDMAOpCode::SEND;
        msg->rdma_hdr.conn_id = conn_id_;
        msg->rdma_hdr.server = false;
        msg->rdma_hdr.num_pkts = 1;
        msg->rdma_hdr.num_segs = 1;
        msg->rdma_hdr.wr_id = wr_id++;
        strcpy(msg->rdma_hdr.local_mr_name, "DATA_TX_CPU_CLIENT");      

        while ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {}

        // Set the length the same as the buffer size
        set_packet_lengths(msg, 0, {message_size_.get()});
        
        HOLOSCAN_LOG_INFO("Sending burst to server with client cmid {}", (void*)conn_id_);      
        send_tx_burst(msg);
        sent = true;
        sleep(1);
      }
    }
  }

 private:
  bool sent = false;
  uint64_t wr_id = 0x1234;
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  uint32_t server_addr_;
  uintptr_t conn_id_ = 0;
  Parameter<bool> rdma_write_;               // Message size in bytes
  Parameter<uint32_t> message_size_;               // Message size in bytes
  Parameter<std::string> server_addr_str_;         // Server address
  Parameter<std::string> client_addr_str_;         // Client address
  Parameter<uint16_t> server_port_;              // Server port
  Parameter<bool> writes_;                        // Writes
  Parameter<bool> reads_;                         // Reads
};

}  // namespace holoscan::ops
