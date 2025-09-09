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

#include <unistd.h>
#include <algorithm>
#include <string>
#include "basic_network_operator_tx.h"

namespace holoscan::ops {

void BasicNetworkOpTx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in");

  spec.param<std::string>(
      ip_addr_, "ip_addr", "IP Address", "IP address of interface to bind to", "0.0.0.0");
  spec.param<uint16_t>(port_, "dst_port", "L4 port ", "UDP or TCP port to listen on");
  spec.param<std::string>(
      l4_proto_p_, "l4_proto", "Layer 4 protocol", "Layer 4 protocol (udp or tcp)", "udp");
  spec.param<uint16_t>(
      max_payload_size_, "max_payload_size", "Max payload size", "Largest payload size");
  spec.param<uint32_t>(ipg_,
                       "min_ipg_ns",
                       "Minimum inter-packet gap",
                       "Smallest gap between packets in nanoseconds");
  spec.param<int32_t>(retry_connect_,
                       "retry_connect",
                       "Re-connect() interval",
                       "Interval to retry connecting to server in seconds",
                       1);
  spec.param<bool>(
      delete_payload_, "delete_payload", "Delete payload", "Delete payload after sending", true);
}
void BasicNetworkOpTx::initialize() {
  HOLOSCAN_LOG_INFO("BasicNetworkOpTx::initialize()");
  holoscan::Operator::initialize();

  memset(&server_addr_, 0, sizeof(server_addr_));

  if (inet_pton(AF_INET, ip_addr_.get().c_str(), &(server_addr_.sin_addr)) != 1) {
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
  }

  if (ipg_.get() > 0) {
    ts_.tv_sec = 0;
    ts_.tv_nsec = ipg_.get();
  }
}

void BasicNetworkOpTx::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                               [[maybe_unused]] ExecutionContext&) {
  HOLOSCAN_LOG_DEBUG("BasicNetworkOpTx::compute");
  auto msg = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in").value();
  int sent;

  if (!connected_) {
    auto ret = connect(sockfd_, (struct sockaddr*)&server_addr_, sizeof(server_addr_));
    if (ret < 0) {
      if (retry_connect_.get() == -1) {
        HOLOSCAN_LOG_INFO("Failed to connect to TCP server at {}:{}. Retries disabled.",
                          ip_addr_.get(), port_.get());
      }

      if (retry_connect_.get() > 0) {
        HOLOSCAN_LOG_INFO("Failed to connect to TCP server at {}:{}. Trying again in {}s...",
                          ip_addr_.get(), port_.get(), retry_connect_.get());
        sleep(retry_connect_.get());
      }

      return;
    }

    connected_ = true;
    HOLOSCAN_LOG_INFO("Successfully connected to server at {}:{}", ip_addr_.get(), port_.get());
  }

  int packets_sent = 0;
  while (msg->len > 0) {
    auto pkt_size = msg->packet_sizes.empty()
                  ? std::min(msg->len, static_cast<uint32_t>(max_payload_size_.get()))
                  : msg->packet_sizes[packets_sent];
    if (l4_proto_ == L4Proto::UDP) {
      sent = sendto(sockfd_,
                    msg->data + byte_cnt_,
                    static_cast<size_t>(pkt_size),
                    MSG_DONTWAIT,
                    reinterpret_cast<const struct sockaddr*>(&server_addr_),
                    sizeof(server_addr_));

      if (sent == -1) {
        HOLOSCAN_LOG_ERROR("Error while sending UDP packet: {}", errno);
        continue;
      }
    } else if (l4_proto_ == L4Proto::TCP) {
      sent = send(sockfd_,
                  msg->data + byte_cnt_,
                  static_cast<size_t>(pkt_size),
                  MSG_DONTWAIT);

      if (sent == -1) {
        HOLOSCAN_LOG_ERROR("Error while sending TCP packet: {}", errno);
        continue;
      }
    }

    packets_sent++;
    msg->len -= sent;
    byte_cnt_ += sent;

    if (ipg_.get() > 0) { nanosleep(&ts_, nullptr); }
  }

  byte_cnt_ = 0;

  if (delete_payload_.get()) {
    delete[] msg->data;
  }

  HOLOSCAN_LOG_DEBUG("BasicNetworkOpTx::compute done");
}

};  // namespace holoscan::ops
