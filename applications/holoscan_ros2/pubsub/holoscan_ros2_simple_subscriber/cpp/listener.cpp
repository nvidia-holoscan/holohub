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

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <holoscan/ros2/operators/subscriber.hpp>

/**
 * @brief A simple Holoscan operator that subscribes to string messages from ROS2.
 * 
 * This operator demonstrates basic integration between Holoscan and ROS2 by receiving
 * string messages from a ROS2 topic and logging them to the console. It inherits from
 * SubscriberOp to leverage the built-in ROS2 subscription capabilities.
 * 
 * The operator processes incoming messages as they arrive and displays the received
 * message content using Holoscan's logging system.
 */
class SimpleSubscriberOp : public holoscan::ros2::ops::SubscriberOp<std_msgs::msg::String> {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SimpleSubscriberOp,
                                       holoscan::ros2::ops::SubscriberOp<std_msgs::msg::String>)

  SimpleSubscriberOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto message = receive().get();
    HOLOSCAN_LOG_INFO("I heard: '{}'", message.data);
  }
};

class HoloscanSimpleSubscriberApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto bridge = make_resource<holoscan::ros2::Bridge>("holoscan_subscriber_resource",
                                                        "holoscan_subscriber_node");

    // Define the operators
    auto subscriber_op =
        make_operator<SimpleSubscriberOp>("SimpleSubscriberOp",
                                          holoscan::Arg("ros2_bridge", bridge),
                                          holoscan::Arg("topic_name", std::string("topic")),
                                          holoscan::Arg("qos", holoscan::ros2::QoS(10)));

    // Define the one-operator workflow
    add_operator(subscriber_op);
  }
};

int main(int argc, char* argv[]) {
  // Initialize ROS2
  rclcpp::init(argc, argv);
  HoloscanSimpleSubscriberApp app;
  app.run();
  return 0;
}
