/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <chrono>
#include <cstdint>
#include <thread>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <daqiri/daqiri.h>
#include <velodyne_lidar.hpp>

class DaqiriSocketRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DaqiriSocketRxOp);
  DaqiriSocketRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::ops::NetworkPacket>("burst_out");
    spec.param(
        server_address_, "server_address", "Server address", "DAQIRI UDP server address", {});
    spec.param(server_port_, "server_port", "Server port", "DAQIRI UDP server port", {});
  }

  void initialize() override {
    holoscan::Operator::initialize();
    while (daqiri::socket_get_server_conn_id(
               server_address_.get(), server_port_.get(), &conn_id_) != daqiri::Status::SUCCESS) {
      HOLOSCAN_LOG_WARN(
          "Waiting for DAQIRI UDP server {}:{}", server_address_.get(), server_port_.get());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override {
    daqiri::BurstParams* burst = nullptr;
    if (daqiri::get_rx_burst(&burst, conn_id_, true) != daqiri::Status::SUCCESS ||
        burst == nullptr) {
      return;
    }

    const auto packets = daqiri::get_num_packets(burst);
    for (int pkt = 0; pkt < packets; ++pkt) {
      const auto length = daqiri::get_packet_length(burst, pkt);
      const auto* payload =
          reinterpret_cast<const uint8_t*>(daqiri::get_packet_ptr(burst, pkt));
      auto out = std::make_shared<holoscan::ops::NetworkPacket>(payload, length, 1);
      op_output.emit(out, "burst_out");
    }
    daqiri::free_all_packets_and_burst_rx(burst);
  }

 private:
  holoscan::Parameter<std::string> server_address_;
  holoscan::Parameter<int> server_port_;
  uintptr_t conn_id_ = 0;
};

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto net_rx = make_operator<DaqiriSocketRxOp>(
        "network_rx", from_config("network_rx"), make_condition<BooleanCondition>("is_alive"));
    auto velodyne_op = make_operator<holoscan::ops::VelodyneLidarOp>("lidar", from_config("lidar"));
    auto viz_op = make_operator<holoscan::ops::HolovizOp>("holoviz", from_config("holoviz"));

    add_flow(net_rx, velodyne_op, {{"burst_out", "burst_in"}});
    add_flow(velodyne_op, viz_op, {{"cloud_out", "receivers"}});
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Expected a YAML configuration file" << std::endl;
    return 1;
  }

  holoscan::set_log_level(holoscan::LogLevel::INFO);
  if (daqiri::daqiri_init(argv[1]) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to initialize DAQIRI");
    return 1;
  }
  auto app = holoscan::make_application<App>();
  app->config(argv[1]);
  app->run();
  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
