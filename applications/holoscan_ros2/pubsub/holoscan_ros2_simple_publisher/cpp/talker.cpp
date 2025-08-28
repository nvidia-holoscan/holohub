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

#include <chrono>
#include <memory>
#include <string>

#include <holoscan/ros2/operators/publisher.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

/**
 * @brief A simple Holoscan operator that publishes string messages to ROS2.
 * 
 * This operator demonstrates basic integration between Holoscan and ROS2 by periodically
 * publishing incrementing "Hello, world!" messages to a ROS2 topic. It inherits from
 * PublisherOp to leverage the built-in ROS2 publishing capabilities.
 * 
 * The operator runs on a periodic schedule and maintains an internal counter to create
 * unique messages on each execution cycle.
 */
class SimplePublisherOp : public holoscan::ros2::ops::PublisherOp<std_msgs::msg::String> {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SimplePublisherOp,
                                       holoscan::ros2::ops::PublisherOp<std_msgs::msg::String>)

  SimplePublisherOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    HOLOSCAN_LOG_INFO("Publishing: '{}'", message.data);
    publish(message);
  }

 private:
  size_t count_;
};

class HoloscanSimplePublisherApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto ros2_bridge = make_resource<holoscan::ros2::Bridge>("holoscan_publisher_resource",
                                                             "holoscan_publisher_node");

    // Define the operators
    auto simple_publisher_op = make_operator<SimplePublisherOp>(
        "SimplePublisherOp",
        make_condition<PeriodicCondition>("period", std::chrono::milliseconds(500)),
        holoscan::Arg("ros2_bridge", ros2_bridge),
        holoscan::Arg("topic_name", std::string("topic")),
        holoscan::Arg("qos", holoscan::ros2::QoS(10)));

    // Define the one-operator workflow
    add_operator(simple_publisher_op);
  }
};

int main(int argc, char** argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);
  HoloscanSimplePublisherApp app;
  app.run();
  return 0;
}
