/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. ..* SPDX-License-Identifier: Apache-2.0
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

#include <memory>
#include <string>

#include "basic_network_operator_rx.h"

namespace holoscan::ops {

void BasicNetworkOpRx::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<NetworkOpBurstParams>>("burst_out");

  spec.param<std::string>(ip_addr_, "ip_addr", "IP Address", "IP address of interface to bind to");
  spec.param<uint16_t>(port_, "dst_port", "L4 port ", "UDP or TCP port to listen on");
  spec.param<std::string>(
      l4_proto_p_, "l4_proto", "Layer 4 protocol", "Layer 4 protocol (udp or tcp)");
  spec.param<uint32_t>(batch_size_, "batch_size", "Batch size", "Number of packets in batch");
  spec.param<uint16_t>(
      max_payload_size_, "max_payload_size", "Max payload size", "Largest payload size");
  spec.param<uint64_t>(max_burst_interval_,
                       "max_burst_interval",
                       "Max burst interval",
                       "Maximum time interval between bursts (ms)");
}

BasicNetworkOpRx::~BasicNetworkOpRx() {
  HOLOSCAN_LOG_INFO("{} packets left in buffer for RX operator", pkts_in_batch_);
}

void BasicNetworkOpRx::stop() {
  // Clean up allocated packet buffer
  if (pkt_buf != nullptr) {
    delete[] pkt_buf;
    pkt_buf = nullptr;
  }
}

void BasicNetworkOpRx::initialize() {
  HOLOSCAN_LOG_INFO("BasicNetworkOpRx::initialize()");
  holoscan::Operator::initialize();

  memset(&server_addr_, 0, sizeof(server_addr_));

  if (inet_pton(AF_INET, ip_addr_.get().c_str(), &(server_addr_.sin_addr.s_addr)) != 1) {
    HOLOSCAN_LOG_CRITICAL("Failed to convert IP address to numeric format!");
    throw;
  }

  server_addr_.sin_family = AF_INET;
  server_addr_.sin_port = htons(port_.get());

  if (l4_proto_p_.get() == "udp") {
    l4_proto_ = L4Proto::UDP;

    if ((sockfd_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to create UDP socket");
      throw;
    }
  } else {
    l4_proto_ = L4Proto::TCP;

    if ((sockfd_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to create TCP socket");
      throw;
    }

    int opt = 1;
    if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
      HOLOSCAN_LOG_CRITICAL("Failed to set socket options");
      throw;
    }
  }


  if (bind(sockfd_, reinterpret_cast<const struct sockaddr*>(&server_addr_),
      sizeof(server_addr_)) < 0) {
    HOLOSCAN_LOG_CRITICAL("Failed to bind to {}:{}!", ip_addr_.get(), port_.get());
  }

  if (l4_proto_ == L4Proto::TCP) {
    if (listen(sockfd_, 1) < 0) {
        HOLOSCAN_LOG_CRITICAL("Error when listening on TCP port");
        throw;
    }
  } else {
    HOLOSCAN_LOG_INFO("Network RX operator bound to {}:{}", ip_addr_.get(), port_.get());
  }
}

void BasicNetworkOpRx::compute([[maybe_unused]] InputContext&, OutputContext& op_output,
                               [[maybe_unused]] ExecutionContext&) {
  HOLOSCAN_LOG_DEBUG("BasicNetworkOpRx::compute");
  sockaddr_in addr;
  socklen_t from_len;
  from_len = sizeof(addr);

  if (l4_proto_ == L4Proto::TCP && !connected_) {
    HOLOSCAN_LOG_INFO("Waiting for incoming TCP connection on {}:{}", ip_addr_.get(), port_.get());
    if ((tcp_sock_ = accept(sockfd_, (struct sockaddr*)&server_addr_,
                            (socklen_t*)&server_addr_)) < 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to accept incoming TCP connection");
        throw;
    }

    HOLOSCAN_LOG_INFO("Successfully attached to incoming connection");
    connected_ = true;
  }

  if (byte_cnt_ == 0) {
    // Reserve memory and initialize burst time on first packet allocation
    if (pkt_buf == nullptr) {
      pkt_buf = new uint8_t[max_payload_size_.get() * batch_size_.get()];
    }
    burst_start_time_ = std::chrono::steady_clock::now();
  }

  while (pkts_in_batch_ < batch_size_.get()) {
    // If enabled, check if max burst interval is reached
    if (max_burst_interval_.get() > 0) {
      auto current_time = std::chrono::steady_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          current_time - burst_start_time_).count();

      if (elapsed_ms >= static_cast<int64_t>(max_burst_interval_.get())) {
        HOLOSCAN_LOG_INFO("Timeout, emitting {} packets (elapsed time: {} ms)",
                          pkts_in_batch_, elapsed_ms);
        break;
      }
    }

    // Receive packets
    int n;
    if (l4_proto_ == L4Proto::UDP) {
      n = recvfrom(sockfd_,
                  &pkt_buf[byte_cnt_],
                  max_payload_size_.get(),
                  MSG_DONTWAIT,
                  (sockaddr*)&addr,
                  &from_len);
    } else if (l4_proto_ == L4Proto::TCP) {
      n = recv(tcp_sock_,
                  &pkt_buf[byte_cnt_],
                  max_payload_size_.get(),
                  0);
    }

    if (n > 0) {
      byte_cnt_ += n;
      pkts_in_batch_++;
    } else {
      return;
    }
  }

  // Only emit if we have packets to send
  if (pkts_in_batch_ > 0) {
    auto msg = std::make_shared<NetworkOpBurstParams>(pkt_buf, byte_cnt_, pkts_in_batch_);
    byte_cnt_ = 0;
    pkts_in_batch_ = 0;
    pkt_buf = nullptr;

    op_output.emit(msg, "burst_out");
  }
  HOLOSCAN_LOG_DEBUG("BasicNetworkOpRx::compute");
}

};  // namespace holoscan::ops
