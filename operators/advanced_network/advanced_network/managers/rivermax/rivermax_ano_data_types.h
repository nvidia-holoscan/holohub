/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RIVERMAX_ANO_DATA_TYPES_H_
#define RIVERMAX_ANO_DATA_TYPES_H_

#include <chrono>

#include "advanced_network/types.h"

namespace holoscan::advanced_network {

class RivermaxBurst;
/**
 * @brief Interface for a collection of bursts.
 *
 * This interface defines the basic operations for a collection of bursts,
 * including enqueueing, dequeueing, checking the size, and clearing the collection.
 */
class IAnoBurstsCollection {
 public:
  /**
   * @brief Virtual destructor for the IAnoBurstsCollection class.
   */
  virtual ~IAnoBurstsCollection() = default;
 /**
   * @brief Enqueues a burst into the queue.
   *
   * @param burst The burst to put into the queue.
   * @return True if the burst was successfully put into the queue, false otherwise.
   */
  virtual bool enqueue_burst(std::shared_ptr<RivermaxBurst> burst) = 0;

  /**
   * @brief Dequeues a burst from the queue.
   *
   * @return A shared pointer to the burst.
   */
  virtual std::shared_ptr<RivermaxBurst> dequeue_burst() = 0;

  /**
   * @brief Gets the number of available bursts in the queue.
   *
   * @return The number of available bursts.
   */
  virtual size_t available_bursts() = 0;

  /**
   * @brief Checks if the queue is empty.
   *
   * @return True if the queue is empty, false otherwise.
   */
  virtual bool empty() = 0;

  /**
   * @brief Clears the queue.
   */
  virtual void stop() = 0;
};

/**
 * @brief Interface for a generic queue.
 *
 * This interface defines the basic operations for a generic queue, including
 * enqueueing, dequeueing, checking the size, and clearing the queue.
 *
 * @tparam T The type of elements in the queue.
 */
template <typename T>
class QueueInterface {
 public:
  /**
   * @brief Virtual destructor for the QueueInterface class.
   */
  virtual ~QueueInterface() = default;

  /**
   * @brief Enqueues a value into the queue.
   *
   * @param value The value to enqueue.
   */
  virtual void enqueue(const T& value) = 0;

  /**
   * @brief Tries to dequeue a value from the queue.
   *
   * @param value The value to dequeue.
   * @return True if the dequeue was successful, false otherwise.
   */
  virtual bool try_dequeue(T& value) = 0;

  /**
   * @brief Tries to dequeue a value from the queue with a timeout.
   *
   * @param value The value to dequeue.
   * @param timeout The timeout for the dequeue operation.
   * @return True if the dequeue was successful, false otherwise.
   */
  virtual bool try_dequeue(T& value, std::chrono::milliseconds timeout) = 0;

  /**
   * @brief Gets the size of the queue.
   *
   * @return The size of the queue.
   */
  virtual size_t get_size() const = 0;

  /**
   * @brief Clears the queue.
   */
  virtual void clear() = 0;
  /**
   * @brief Stops the queue.
   */
  virtual void stop() = 0;
};

/**
 * @brief A queue for handling bursts of packets.
 *
 * The AnoBurstsQueue class implements the IAnoBurstsCollection interface and
 * provides functionality for managing a queue of bursts of packets. It supports
 * operations such as putting bursts into the queue, getting bursts from the queue,
 * checking the number of available bursts, and clearing the queue.
 */
class AnoBurstsQueue : public IAnoBurstsCollection {
 public:
  /**
   * @brief Constructor for the AnoBurstsQueue class.
   *
   * Initializes the AnoBurstsQueue instance.
   */
  AnoBurstsQueue();

  virtual ~AnoBurstsQueue() = default;

  bool enqueue_burst(std::shared_ptr<RivermaxBurst> burst) override;

  std::shared_ptr<RivermaxBurst> dequeue_burst() override;

  size_t available_bursts() override { return queue_->get_size(); };

  bool empty() override { return queue_->get_size() == 0; };

  void stop() override;

  /**
   * @brief Clears the queue.
   */
  void clear();

 private:
  std::unique_ptr<QueueInterface<std::shared_ptr<RivermaxBurst>>> queue_;
  std::atomic<bool> stop_ = false;
};

enum BurstFlags : uint32_t {
  FLAGS_NONE = 0,
  INFO_PER_PACKET = 1,
  FRAME_BUFFER_IS_OWNED = 2,
};

struct AnoBurstExtendedInfo {
  uint32_t tag;
  uint16_t burst_id;
  bool hds_on;
  uint16_t header_stride_size;
  uint16_t payload_stride_size;
  bool header_on_cpu;
  bool payload_on_cpu;
  uint16_t header_seg_idx;
  uint16_t payload_seg_idx;
};

struct RivermaxPacketExtendedInfo {
  uint32_t flow_tag;
  uint64_t timestamp;
};

struct RivermaxPacketData {
  uint8_t* header_ptr;
  uint8_t* payload_ptr;
  size_t header_length;
  size_t payload_length;
  RivermaxPacketExtendedInfo extended_info;
};

/**
 * @brief Class representing Rivermax log levels.
 */
class RivermaxLogLevel {
 public:
  enum Level {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL,
    OFF,
  };

  /**
   * @brief Converts an Rivermax log level to a description string.
   *
   * @param level The Rivermax log level to convert.
   * @return The string representation of the log level.
   */
  static std::string to_description_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<0>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }

  /**
   * @brief Converts an Rivermax log level to a command string.
   *
   * @param level The Rivermax log level to convert.
   * @return The string representation of the log level.
   */
  static std::string to_cmd_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<1>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }
  /**
   * @brief Converts an advanced_network log level to an Rmax log level.
   *
   * @param level The advanced_network log level to convert.
   * @return The corresponding Rmax log level.
   */
  static Level from_adv_net_log_level(LogLevel::Level level) {
    auto it = adv_net_to_rivermax_log_level_map.find(level);
    if (it != adv_net_to_rivermax_log_level_map.end()) { return it->second; }
    return OFF;
  }

 private:
  /**
   * A map of log level to a tuple of the description and command strings.
   */
  static const std::unordered_map<Level, std::tuple<std::string, std::string>> level_to_cmd_map;
  /**
   * A map of advanced_network log level to Rivermax log level.
   */
  static const std::unordered_map<LogLevel::Level, Level> adv_net_to_rivermax_log_level_map;
};

}  // namespace holoscan::advanced_network

#endif  // RIVERMAX_ANO_DATA_TYPES_H_
