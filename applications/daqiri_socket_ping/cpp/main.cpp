/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <daqiri/daqiri.h>
#include "yaml-cpp/yaml.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

static constexpr int NUM_MSGS = 10;
static constexpr int POST_PING_FLUSH_MSGS = 128;

namespace holoscan::ops {

class DaqiriSocketPingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DaqiriSocketPingTxOp)

  DaqiriSocketPingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_addr_str_, "server_address", "Server IP address", "Server IP address", "127.0.0.1");
    spec.param<std::string>(
        client_addr_str_, "client_address", "Client IP address", "Client IP address", "127.0.0.1");
    spec.param<uint16_t>(server_port_, "server_port", "Server port", "Server port", 5001);
  }

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {
    ensure_connected();

    auto* msg = daqiri::create_tx_burst_params();
    daqiri::set_header(msg, port_, queue_, 1, 1);

    while (daqiri::get_tx_packet_burst(msg) != daqiri::Status::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto* payload = reinterpret_cast<uint8_t*>(daqiri::get_packet_ptr(msg, 0));
    const auto value = index_ < NUM_MSGS ? index_ : -1;
    std::memcpy(payload, &value, sizeof(value));
    daqiri::set_packet_lengths(msg, 0, {static_cast<int>(sizeof(value))});
    msg->rdma_hdr.conn_id = conn_id_;

    while (daqiri::send_tx_burst(msg) != daqiri::Status::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (value >= 0) { HOLOSCAN_LOG_INFO("Ping message sent with value {}", value); }
    index_++;
  }

 private:
  void ensure_connected() {
    while (conn_id_ == 0) {
      const auto res = daqiri::socket_connect_to_server(
          server_addr_str_.get(), server_port_.get(), client_addr_str_.get(), &conn_id_);
      if (res != daqiri::Status::SUCCESS) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      if (daqiri::socket_get_port_queue(conn_id_, &port_, &queue_) == daqiri::Status::SUCCESS) {
        HOLOSCAN_LOG_INFO(
            "Connected to server at {}:{}", server_addr_str_.get(), server_port_.get());
        return;
      }

      conn_id_ = 0;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  int index_ = 0;
  uintptr_t conn_id_ = 0;
  uint16_t port_ = 0;
  uint16_t queue_ = 0;
  Parameter<std::string> server_addr_str_;
  Parameter<std::string> client_addr_str_;
  Parameter<uint16_t> server_port_;
};

class DaqiriSocketPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DaqiriSocketPingRxOp)

  DaqiriSocketPingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_addr_str_, "server_address", "Server IP address", "Server IP address", "127.0.0.1");
    spec.param<uint16_t>(server_port_, "server_port", "Server port", "Server port", 5001);
  }

  void compute(InputContext&, OutputContext&, ExecutionContext& context) override {
    if (!ensure_connected()) { return; }

    daqiri::BurstParams* burst = nullptr;
    if (daqiri::get_rx_burst(&burst, conn_id_, true) != daqiri::Status::SUCCESS ||
        burst == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return;
    }

    const auto num_packets = daqiri::get_num_packets(burst);
    for (int pkt = 0; pkt < num_packets; ++pkt) {
      auto* payload = reinterpret_cast<int*>(daqiri::get_packet_ptr(burst, pkt));
      const auto val = *payload;
      if (val < 0) { continue; }
      HOLOSCAN_LOG_INFO("Ping message received with value {}", val);
      if (val == NUM_MSGS - 1) { GxfGraphInterrupt(context.context()); }
    }

    daqiri::free_all_packets_and_burst_rx(burst);
  }

 private:
  bool ensure_connected() {
    if (conn_id_ != 0) { return true; }

    const auto ret =
        daqiri::socket_get_server_conn_id(server_addr_str_.get(), server_port_.get(), &conn_id_);
    if (ret != daqiri::Status::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      return false;
    }

    if (daqiri::socket_get_port_queue(conn_id_, &port_, &queue_) != daqiri::Status::SUCCESS) {
      conn_id_ = 0;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      return false;
    }

    HOLOSCAN_LOG_INFO(
        "Accepted client connection on {}:{}", server_addr_str_.get(), server_port_.get());
    return true;
  }

  uintptr_t conn_id_ = 0;
  uint16_t port_ = 0;
  uint16_t queue_ = 0;
  Parameter<std::string> server_addr_str_;
  Parameter<uint16_t> server_port_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  explicit App(bool rx_enabled, bool tx_enabled)
      : rx_enabled_(rx_enabled), tx_enabled_(tx_enabled) {}

  void compose() override {
    using namespace holoscan;

    if (tx_enabled_) {
      HOLOSCAN_LOG_INFO("Transmit enabled");
      auto tx = make_operator<ops::DaqiriSocketPingTxOp>(
          "ping_tx",
          from_config("ping_tx"),
          make_condition<CountCondition>(NUM_MSGS + POST_PING_FLUSH_MSGS));
      add_operator(tx);
    }

    if (rx_enabled_) {
      HOLOSCAN_LOG_INFO("Receive enabled");
      auto rx = make_operator<ops::DaqiriSocketPingRxOp>(
          "ping_rx", from_config("ping_rx"), make_condition<BooleanCondition>("is_alive"));
      add_operator(rx);
    }
  }

 private:
  bool rx_enabled_ = false;
  bool tx_enabled_ = false;
};

bool config_has_map(const std::string& config_path, const char* key) {
  const auto root = YAML::LoadFile(config_path);
  const auto node = root[key];
  return node && node.IsMap();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

  const auto config_path = std::string(argv[1]);
  if (daqiri::daqiri_init(config_path) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to configure DAQIRI");
    return -1;
  }

  const bool rx_enabled = config_has_map(config_path, "ping_rx");
  const bool tx_enabled = config_has_map(config_path, "ping_tx");
  if (!rx_enabled && !tx_enabled) {
    HOLOSCAN_LOG_ERROR("Configuration must contain ping_rx or ping_tx");
    daqiri::shutdown();
    return -1;
  }

  auto app = holoscan::make_application<App>(rx_enabled, tx_enabled);
  app->config(config_path);
  app->run();

  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
