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

#ifndef HOLOSCAN_ROS2_BRIDGE_HPP
#define HOLOSCAN_ROS2_BRIDGE_HPP

#include <future>
#include <mutex>
#include <queue>

#include <holoscan/holoscan.hpp>
#include <holoscan/ros2/yaml_converter.hpp>
#include <rclcpp/rclcpp.hpp>

namespace holoscan::ros2 {

class Bridge : public holoscan::Resource {
 public:
  using SharedPtr = std::shared_ptr<Bridge>;

  Bridge() = default;  // to satisfy the yaml converter
  explicit Bridge(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {}

  explicit Bridge(const std::string& node_name,
                  const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : node_(std::make_shared<rclcpp::Node>(node_name, options)) {}
  Bridge(const std::string& node_name, const std::string& namespace_name,
         const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : node_(std::make_shared<rclcpp::Node>(node_name, namespace_name, options)) {}

  // Explicitly default the copy constructor and assignment operator
  Bridge(Bridge&& other) noexcept = default;
  Bridge& operator=(Bridge&& other) noexcept = default;

  ~Bridge() {
    HOLOSCAN_LOG_DEBUG("Shutting down ROS2 bridge");
    if (spin_future_.valid())
      rclcpp::shutdown();
    HOLOSCAN_LOG_DEBUG("ROS2 bridge shut down");
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("Initializing ROS2 bridge");
    if (!spin_future_.valid()) {
      spin_future_ = std::async(
          std::launch::async,
          [](rclcpp::Node::SharedPtr node) {
            HOLOSCAN_LOG_DEBUG("Starting ROS2 spin");
            rclcpp::spin(node);
            HOLOSCAN_LOG_DEBUG("ROS2 spin ended");
          },
          node_);
    }
  }

  bool valid() const { return static_cast<bool>(node_); }

  template <typename MessageT>
  class Publisher {
   public:
    using MessageType = MessageT;
    using SharedPtr = typename std::shared_ptr<Publisher<MessageType>>;
    Publisher() = default;
    Publisher(rclcpp::Node::SharedPtr node, const std::string& topic_name, const rclcpp::QoS& qos)
        : publisher_(node->create_publisher<MessageType>(topic_name, qos)) {}

    void publish(const MessageType& message) { publisher_->publish(message); }

   private:
    typename rclcpp::Publisher<MessageType>::SharedPtr publisher_;
  };

  template <typename MessageT>
  std::shared_ptr<Publisher<MessageT>> create_publisher(const std::string& topic_name,
                                                        const rclcpp::QoS& qos) {
    return std::make_shared<Publisher<MessageT>>(node_, topic_name, qos);
  }

  template <typename MessageT, template <typename> class ContainerT>
  class Subscriber {
   public:
    using MessageType = MessageT;
    using SharedPtr = typename std::shared_ptr<Subscriber<MessageT, ContainerT>>;
    using MessageQueue = std::queue<MessageType, ContainerT<MessageType>>;
    Subscriber() = default;
    Subscriber(rclcpp::Node::SharedPtr node, const std::string& topic_name, const rclcpp::QoS& qos,
               typename MessageQueue::size_type message_queue_max_size = 0)
        : subscriber_(node->create_subscription<MessageType>(
              topic_name, qos, std::bind(&Subscriber::on_receive, this, std::placeholders::_1))),
          message_queue_max_size_(message_queue_max_size) {}

    std::future<MessageType> receive() {
      std::lock_guard<std::mutex> lock(mutex_);
      std::promise<MessageType> promise;
      auto future = promise.get_future();
      if (message_queue_.empty()) {
        promise_queue_.emplace(std::move(promise));
        return future;
      }
      promise.set_value(message_queue_.front());
      message_queue_.pop();
      return future;
    }

   private:
    using PromiseQueue =
        std::queue<std::promise<MessageType>, ContainerT<std::promise<MessageType>>>;
    typename rclcpp::Subscription<MessageType>::SharedPtr subscriber_;

    std::mutex mutex_;
    MessageQueue message_queue_;
    PromiseQueue promise_queue_;
    typename MessageQueue::size_type message_queue_max_size_;

    void on_receive(const MessageType& message) {
      HOLOSCAN_LOG_TRACE("On receive");
      std::lock_guard<std::mutex> lock(mutex_);
      if (promise_queue_.empty()) {  // No waiting promises
        if (message_queue_max_size_ == 0 || message_queue_.size() < message_queue_max_size_) {
          message_queue_.push(message);
        } else {
          HOLOSCAN_LOG_WARN("Message queue is full, dropping message");
        }
      } else {
        promise_queue_.front().set_value(message);
        promise_queue_.pop();
      }
    }
  };

  template <typename MessageT, template <typename> class ContainerT = std::deque>
  std::shared_ptr<Subscriber<MessageT, ContainerT>> create_subscription(
      const std::string& topic_name, const rclcpp::QoS& qos) {
    return std::make_shared<Subscriber<MessageT, ContainerT>>(node_, topic_name, qos);
  }

 private:
  rclcpp::Node::SharedPtr node_;
  std::shared_future<void> spin_future_;
};

using BridgePtr = Bridge::SharedPtr;

}  // namespace holoscan::ros2

ROS2_DECLARE_YAML_CONVERTER_UNSUPPORTED(holoscan::ros2::Bridge)  // NOLINT

#endif /* HOLOSCAN_ROS2_BRIDGE_HPP */
