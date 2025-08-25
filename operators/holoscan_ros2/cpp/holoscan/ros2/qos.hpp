/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_ROS2_QOS_HPP
#define HOLOSCAN_ROS2_QOS_HPP

#include "holoscan/ros2/yaml_converter.hpp"
#include "rclcpp/qos.hpp"

namespace holoscan::ros2 {

class QoS : public rclcpp::QoS {
 public:
  QoS()
      : rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default)) {
  }  // default constructor
  using rclcpp::QoS::QoS;  // constructor from rclcpp::QoS
};

}  // namespace holoscan::ros2

ROS2_DECLARE_YAML_CONVERTER_UNSUPPORTED(holoscan::ros2::QoS)

#endif /* HOLOSCAN_ROS2_QOS_HPP */
