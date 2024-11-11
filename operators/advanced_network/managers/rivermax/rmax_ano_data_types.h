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

#ifndef RMAX_ANO_DATA_TYPES_H_
#define RMAX_ANO_DATA_TYPES_H_

#include "adv_network_types.h"


namespace holoscan::ops {

// using namespace holoscan::ops;
class RmaxBurst;

class IAnoBurstsCollection {
 public:
  virtual ~IAnoBurstsCollection() = default;
  virtual bool enqueue_burst(std::shared_ptr<RmaxBurst> burst) = 0;
  virtual std::shared_ptr<RmaxBurst> dequeue_burst() = 0;
  virtual size_t available_bursts() = 0;
  virtual bool empty() = 0;
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

  /**
   * @brief Virtual destructor for the AnoBurstsQueue class.
   */
  virtual ~AnoBurstsQueue() = default;

  /**
   * @brief Enqueues a burst into the queue.
   *
   * @param burst The burst to put into the queue.
   * @return True if the burst was successfully put into the queue, false otherwise.
   */
  bool enqueue_burst(std::shared_ptr<RmaxBurst> burst) override;

  /**
   * @brief Dequeues a burst from the queue.
   *
   * @return A shared pointer to the burst.
   */
  std::shared_ptr<RmaxBurst> dequeue_burst() override;

  /**
   * @brief Gets the number of available bursts in the queue.
   *
   * @return The number of available bursts.
   */
  size_t available_bursts() override { return m_queue->get_size(); };

  /**
   * @brief Checks if the queue is empty.
   *
   * @return True if the queue is empty, false otherwise.
   */
  bool empty() override { return m_queue->get_size() == 0; };

  /**
   * @brief Clears the queue.
   */
  void clear();

 private:
  std::unique_ptr<QueueInterface<std::shared_ptr<RmaxBurst>>> m_queue;
};

enum BurstFlags : uint8_t {
  FLAGS_NONE = 0,
  INFO_PER_PACKET = 1,
};

struct AnoBurstExtendedInfo {
  uint32_t tag;
  BurstFlags burst_flags;
  uint16_t burst_id;
  bool hds_on;
  uint16_t header_stride_size;
  uint16_t payload_stride_size;
  bool header_on_cpu;
  bool payload_on_cpu;
  uint16_t header_seg_idx;
  uint16_t payload_seg_idx;
};

struct RmaxPacketExtendedInfo {
  uint32_t flow_tag;
  uint64_t timestamp;
};

struct RmaxPacketData {
  uint8_t* header_ptr;
  uint8_t* payload_ptr;
  size_t header_length;
  size_t payload_length;
  RmaxPacketExtendedInfo extended_info;
};

/**
 * @brief Class representing Rmax log levels.
 */
class RmaxLogLevel {
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
   * @brief Converts an Rmax log level to a description string.
   *
   * @param level The Rmax log level to convert.
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
   * @brief Converts an Rmax log level to a command string.
   *
   * @param level The Rmax log level to convert.
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
   * @brief Converts an Ano log level to an Rmax log level.
   *
   * @param ano_level The Ano log level to convert.
   * @return The corresponding Rmax log level.
   */
  static Level from_ano_log_level(AnoLogLevel::Level ano_level) {
    auto it = ano_to_rmax_log_level_map.find(ano_level);
    if (it != ano_to_rmax_log_level_map.end()) { return it->second; }
    return OFF;
  }

 private:
  /**
   * A map of log level to a tuple of the description and command strings.
   */
  static const std::unordered_map<Level, std::tuple<std::string, std::string>> level_to_cmd_map;
  /**
   * A map of Ano log level to Rmax log level.
   */
  static const std::unordered_map<AnoLogLevel::Level, Level> ano_to_rmax_log_level_map;
};

}  // namespace holoscan::ops

#endif  // RMAX_ANO_DATA_TYPES_H_
