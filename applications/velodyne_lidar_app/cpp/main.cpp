/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <basic_network_operator_rx.h>
#include <velodyne_lidar.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto net_rx = make_operator<ops::BasicNetworkOpRx>(
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
  auto app = holoscan::make_application<App>();
  app->config(argv[1]);
  app->run();
  return 0;
}
