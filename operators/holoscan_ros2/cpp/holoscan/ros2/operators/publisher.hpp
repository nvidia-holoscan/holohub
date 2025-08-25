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

#ifndef HOLOSCAN_ROS2_OPERATORS_PUBLISHER_HPP
#define HOLOSCAN_ROS2_OPERATORS_PUBLISHER_HPP

#include <holoscan/ros2/operator.hpp>
#include <holoscan/ros2/qos.hpp>

namespace holoscan::ros2::ops {

/**
 * @brief ROS2 Publisher Operator for Holoscan applications
 *
 * This operator provides a bridge between Holoscan dataflow and ROS2 topics by publishing
 * messages from Holoscan operators to ROS2 topics.
 *
 * @tparam MessageT The ROS2 message type to publish (e.g., std_msgs::msg::String)
 *
 * ## Usage:
 *
 * The typical usage pattern is to inherit from PublisherOp and override the compute() function
 * to create and publish messages:
 *
 * ```cpp
 * class SimplePublisherOp : public holoscan::ros2::ops::PublisherOp<std_msgs::msg::String> {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SimplePublisherOp,
 *                                        holoscan::ros2::ops::PublisherOp<std_msgs::msg::String>)
 *
 *   void compute(holoscan::InputContext& op_input,
 *                holoscan::OutputContext& op_output,
 *                holoscan::ExecutionContext& context) override {
 *     auto message = std_msgs::msg::String();
 *     // Set your message data here
 *     publish(message);  // Publish the message to the ROS2 topic
 *   }
 * };
 *
 * // In your application:
 * auto publisher_op = make_operator<SimplePublisherOp>(
 *     "SimplePublisherOp",
 *     holoscan::Arg("ros2_bridge", bridge),
 *     holoscan::Arg("topic_name", std::string("topic")),
 *     holoscan::Arg("qos", holoscan::ros2::QoS(10))
 * );
 * ```
 *
 * ## Parameters:
 * - **ros2_bridge**: (Required) Shared ROS2 bridge resource
 * - **topic_name**: (Required) ROS2 topic name to publish to
 * - **qos**: (Optional) QoS settings for the publisher (defaults to default QoS)
 *
 * ## Methods:
 * - **publish(message)**: Publishes a message to the ROS2 topic
 *
 * ## Notes:
 * - Inherit from this class and override compute() to create and publish messages
 * - Call publish() in your compute() function to send messages to the ROS2 topic
 * - The operator automatically handles ROS2 node lifecycle and publisher management
 */
template <typename MessageT>
class PublisherOp : public holoscan::ros2::Operator {
 public:
  using MessageType = MessageT;
  using Publisher = typename holoscan::ros2::Bridge::Publisher<MessageType>;
  using PublisherPtr = typename Publisher::SharedPtr;
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PublisherOp, holoscan::ros2::Operator)

  PublisherOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    /// Register converters for arguments not defined by Holoscan
    register_converter<QoS>();

    spec.param(topic_name_, "topic_name", "topic_name", "Topic name to publish messages to");
    spec.param(qos_, "qos", "qos", "QoS for the publisher", QoS());

    holoscan::ros2::Operator::setup(spec);
  }

  void initialize() override {
    holoscan::ros2::Operator::initialize();
    publisher_ = ros2_bridge()->create_publisher<MessageType>(topic_name_.get(), qos_.get());
  }

  void publish(const MessageType& message) { publisher_->publish(message); }

 private:
  holoscan::Parameter<std::string> topic_name_;
  holoscan::Parameter<QoS> qos_;

  PublisherPtr publisher_;
};

}  // namespace holoscan::ros2::ops

#endif /* HOLOSCAN_ROS2_OPERATORS_PUBLISHER_HPP */
