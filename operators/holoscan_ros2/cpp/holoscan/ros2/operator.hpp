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

#ifndef HOLOSCAN_ROS2_OPERATOR_HPP
#define HOLOSCAN_ROS2_OPERATOR_HPP

#include <holoscan/core/operator.hpp>
#include <holoscan/ros2/bridge.hpp>

namespace holoscan::ros2 {

class Operator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(Operator, holoscan::Operator)

  Operator() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    /// Add parameters to the operator spec
    spec.param(ros2_bridge_, "ros2_bridge", "ROS2Bridge", "ROS2 bridge object");

    holoscan::Operator::setup(spec);
  }

  void initialize() override {
    holoscan::Operator::initialize();
    assert(ros2_bridge_.get());
    assert(ros2_bridge_.get()->valid());
  }

 protected:
  BridgePtr ros2_bridge() { return ros2_bridge_.get(); }

 private:
  holoscan::Parameter<BridgePtr> ros2_bridge_;
};

}  // namespace holoscan::ros2

#endif /* HOLOSCAN_ROS2_OPERATOR_HPP */
