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

#ifndef HOLOSCAN_ROS2_OPERATORS_SUBSCRIBER_HPP
#define HOLOSCAN_ROS2_OPERATORS_SUBSCRIBER_HPP

#include <holoscan/ros2/operator.hpp>

#include <future>
#include <mutex>
#include <queue>

#include <holoscan/ros2/qos.hpp>

namespace holoscan::ros2::ops {

/**
 * @brief ROS2 Subscriber Operator for Holoscan applications
 *
 * This operator provides a bridge between ROS2 topics and Holoscan dataflow by subscribing
 * to ROS2 messages and making them available to Holoscan operators.
 *
 * @tparam MessageT The ROS2 message type to subscribe to (e.g., std_msgs::msg::String)
 * @tparam ContainerT The container type for storing messages (defaults to std::deque)
 *
 * ## Usage:
 *
 * The typical usage pattern is to inherit from SubscriberOp and override the compute() function
 * to process received messages:
 *
 * ```cpp
 * class SimpleSubscriberOp : public holoscan::ros2::ops::SubscriberOp<std_msgs::msg::String> {
 *  public:
 *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SimpleSubscriberOp,
 *                                        holoscan::ros2::ops::SubscriberOp<std_msgs::msg::String>)
 *
 *   void compute(holoscan::InputContext& op_input,
 *                holoscan::OutputContext& op_output,
 *                holoscan::ExecutionContext& context) override {
 *     auto message = receive().get();  // Get the latest message from the ROS2 topic
 *   }
 * };
 *
 * // In your application:
 * auto subscriber_op = make_operator<SimpleSubscriberOp>(
 *     "SimpleSubscriberOp",
 *     holoscan::Arg("ros2_bridge", bridge),
 *     holoscan::Arg("topic_name", std::string("topic")),
 *     holoscan::Arg("qos", holoscan::ros2::QoS(10))
 * );
 * ```
 *
 * ## Parameters:
 * - **ros2_bridge**: (Required) Shared ROS2 bridge resource
 * - **topic_name**: (Required) ROS2 topic name to subscribe to
 * - **qos**: (Optional) QoS settings for the subscription (defaults to default QoS)
 * - **message_queue_max_size**: (Optional) Maximum messages to buffer (0 = unlimited)
 *
 * ## Methods:
 * - **receive()**: Returns a std::future<MessageType> for asynchronous message retrieval
 *
 * ## Notes:
 * - Inherit from this class and override compute() to process messages
 * - Call receive() in your compute() function to get the latest message
 * - The operator automatically handles ROS2 node lifecycle and subscription management
 * - Messages are buffered internally and can be retrieved asynchronously
 */
template <typename MessageT, template <typename> class ContainerT = std::deque>
class SubscriberOp : public holoscan::ros2::Operator {
 public:
  using MessageType = MessageT;
  using Subscriber = typename holoscan::ros2::Bridge::Subscriber<MessageType, ContainerT>;
  using SubscriberPtr = typename Subscriber::SharedPtr;

  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(SubscriberOp, holoscan::ros2::Operator)

  SubscriberOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    /// Register converters for arguments not defined by Holoscan
    register_converter<QoS>();

    spec.param(topic_name_, "topic_name", "topic_name", "Topic name to subscribe to");
    spec.param(qos_, "qos", "qos", "QoS for the subscriber", QoS());
    spec.param(message_queue_max_size_,
               "message_queue_max_size",
               "message_queue_max_size",
               "Maximum number of messages to store in the message queue",
               typename Subscriber::MessageQueue::size_type(0));

    holoscan::ros2::Operator::setup(spec);
  }

  void initialize() override {
    holoscan::ros2::Operator::initialize();
    subscriber_ = ros2_bridge()->create_subscription<MessageType>(topic_name_.get(), qos_.get());
  }

  std::future<MessageType> receive() { return subscriber_->receive(); }

 private:
  holoscan::Parameter<std::string> topic_name_;
  holoscan::Parameter<QoS> qos_;
  holoscan::Parameter<typename Subscriber::MessageQueue::size_type> message_queue_max_size_;

  SubscriberPtr subscriber_;
};

}  // namespace holoscan::ros2::ops
#endif /* HOLOSCAN_ROS2_OPERATORS_SUBSCRIBER_HPP */
