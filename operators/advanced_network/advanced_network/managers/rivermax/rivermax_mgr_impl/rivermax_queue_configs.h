/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RIVERMAX_QUEUE_CONFIGS_H_
#define RIVERMAX_QUEUE_CONFIGS_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "rdk/rivermax_dev_kit.h"
#include "rdk/apps/rmax_ipo_receiver/rmax_ipo_receiver.h"
#include "rdk/apps/rmax_rtp_receiver/rmax_rtp_receiver.h"
#include "rdk/apps/rmax_xstream_media_sender/rmax_xstream_media_sender.h"

#include "advanced_network/manager.h"
#include "rivermax_ano_data_types.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps::rmax_ipo_receiver;
using namespace rivermax::dev_kit::apps::rmax_rtp_receiver;
using namespace rivermax::dev_kit::apps::rmax_xstream_media_sender;

enum class QueueConfigType { IPOReceiver, RTPReceiver, MediaFrameSender, GenericPacketSender };

/** @brief Converts QueueConfigType to string.
 *
 * This function converts the QueueConfigType enum value to its corresponding
 * string representation.
 *
 * @param type The QueueConfigType enum value to convert.
 * @return The string representation of the QueueConfigType.
 */
std::string queue_config_type_to_string(QueueConfigType type);

/** @brief Base class for additional queue configuration.
 *
 * This class serves as a base for all additional queue configurations.
 */
class BaseQueueConfig : public ManagerExtraQueueConfig {
 public:
  virtual ~BaseQueueConfig() = default;
  /**
   * @brief Dumps the additional queue configuration.
   *
   * This function must be implemented by derived classes to dump the additional
   * queue configuration to the log.
   */
  virtual void dump_parameters() const {}
  /**
   * @brief Returns the type of the queue configuration.
   *
   * This function must be implemented by derived classes to return the type of
   * the queue configuration.
   *
   * @return The type of the queue configuration.
   */
  virtual QueueConfigType get_type() const = 0;
};

/**
 * @brief Configuration structure for Rivermax RX queue.
 *
 * This structure holds the configuration settings for an Rivermax RX queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RivermaxCommonRxQueueConfig : public BaseQueueConfig {
 public:
  RivermaxCommonRxQueueConfig() = default;
  virtual ~RivermaxCommonRxQueueConfig() = default;

  RivermaxCommonRxQueueConfig(const RivermaxCommonRxQueueConfig& other);
  RivermaxCommonRxQueueConfig& operator=(const RivermaxCommonRxQueueConfig& other);

 public:
  uint16_t max_packet_size = 0;
  size_t max_chunk_size;
  size_t packets_buffers_size;
  bool gpu_direct;
  int gpu_device_id;
  bool lock_gpu_clocks;
  uint16_t split_boundary;
  size_t num_of_threads;
  bool print_parameters;
  int sleep_between_operations_us;
  std::string allocator_type;
  bool ext_seq_num;
  bool memory_registration;
  bool send_packet_ext_info;
  uint32_t stats_report_interval_ms;
  std::string cpu_cores;
  int master_core;

  // Burst pool adaptive dropping configuration
  bool burst_pool_adaptive_dropping_enabled = false;
  uint32_t burst_pool_low_threshold_percent = 25;
  uint32_t burst_pool_critical_threshold_percent = 10;
  uint32_t burst_pool_recovery_threshold_percent = 50;
};

/**
 * @brief Configuration structure for Rivermax IPO receiver queue.
 *
 * This structure holds the configuration settings for an Rivermax IPO receiver queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RivermaxIPOReceiverQueueConfig : public RivermaxCommonRxQueueConfig {
 public:
  RivermaxIPOReceiverQueueConfig() = default;
  virtual ~RivermaxIPOReceiverQueueConfig() = default;

  RivermaxIPOReceiverQueueConfig(const RivermaxIPOReceiverQueueConfig& other);
  RivermaxIPOReceiverQueueConfig& operator=(const RivermaxIPOReceiverQueueConfig& other);

  QueueConfigType get_type() const override { return QueueConfigType::IPOReceiver; }
  void dump_parameters() const override;

 public:
  std::vector<std::string> local_ips;
  std::vector<std::string> source_ips;
  std::vector<std::string> destination_ips;
  std::vector<uint16_t> destination_ports;
  uint32_t max_path_differential_us;
};

/**
 * @brief Configuration structure for Rivermax RTP receiver queue.
 *
 * This structure holds the configuration settings for an Rivermax RTP receiver queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RivermaxRTPReceiverQueueConfig : public RivermaxCommonRxQueueConfig {
 public:
  RivermaxRTPReceiverQueueConfig() = default;
  virtual ~RivermaxRTPReceiverQueueConfig() = default;

  RivermaxRTPReceiverQueueConfig(const RivermaxRTPReceiverQueueConfig& other);
  RivermaxRTPReceiverQueueConfig& operator=(const RivermaxRTPReceiverQueueConfig& other);

  QueueConfigType get_type() const override { return QueueConfigType::RTPReceiver; }
  void dump_parameters() const override;

 public:
  std::string local_ip;
  std::string source_ip;
  std::string destination_ip;
  uint16_t destination_port;
};

/**
 * @brief Configuration structure for Rivermax TX queue.
 *
 * This structure holds the configuration settings for an Rivermax TX queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RivermaxCommonTxQueueConfig : public BaseQueueConfig {
 public:
  RivermaxCommonTxQueueConfig() = default;
  virtual ~RivermaxCommonTxQueueConfig() = default;

  RivermaxCommonTxQueueConfig(const RivermaxCommonTxQueueConfig& other);
  RivermaxCommonTxQueueConfig& operator=(const RivermaxCommonTxQueueConfig& other);

 public:
  bool gpu_direct;
  int gpu_device_id;
  bool lock_gpu_clocks;
  uint16_t split_boundary;
  std::string local_ip;
  std::string destination_ip;
  uint16_t destination_port;
  size_t num_of_threads;
  bool print_parameters;
  bool sleep_between_operations;
  std::string allocator_type;
  bool memory_allocation;
  bool memory_registration;
  bool send_packet_ext_info;
  size_t num_of_packets_in_chunk;
  uint32_t stats_report_interval_ms;
  std::string cpu_cores;
  int master_core;
  bool dummy_sender;
};

struct RivermaxMediaSenderQueueConfig : public RivermaxCommonTxQueueConfig {
 public:
  RivermaxMediaSenderQueueConfig() = default;
  virtual ~RivermaxMediaSenderQueueConfig() = default;

  RivermaxMediaSenderQueueConfig(const RivermaxMediaSenderQueueConfig& other);
  RivermaxMediaSenderQueueConfig& operator=(const RivermaxMediaSenderQueueConfig& other);

  QueueConfigType get_type() const override { return QueueConfigType::MediaFrameSender; }
  void dump_parameters() const override;

 public:
  std::string video_format;
  uint16_t bit_depth;
  uint16_t frame_width;
  uint16_t frame_height;
  uint16_t frame_rate;
  bool use_internal_memory_pool;
  MemoryKind memory_pool_location;
};

class RivermaxCommonRxQueueValidator : public ISettingsValidator<RivermaxCommonRxQueueConfig> {
 public:
  ReturnStatus validate(
      const std::shared_ptr<RivermaxCommonRxQueueConfig>& settings) const override;
};

class RivermaxIPOReceiverQueueValidator
    : public ISettingsValidator<RivermaxIPOReceiverQueueConfig> {
 public:
  ReturnStatus validate(
      const std::shared_ptr<RivermaxIPOReceiverQueueConfig>& settings) const override;
};

class RivermaxRTPReceiverQueueValidator
    : public ISettingsValidator<RivermaxRTPReceiverQueueConfig> {
 public:
  ReturnStatus validate(
      const std::shared_ptr<RivermaxRTPReceiverQueueConfig>& settings) const override;
};

class RivermaxCommonTxQueueValidator : public ISettingsValidator<RivermaxCommonTxQueueConfig> {
 public:
  ReturnStatus validate(
      const std::shared_ptr<RivermaxCommonTxQueueConfig>& settings) const override;
};

class RivermaxMediaSenderQueueValidator
    : public ISettingsValidator<RivermaxMediaSenderQueueConfig> {
 public:
  ReturnStatus validate(
      const std::shared_ptr<RivermaxMediaSenderQueueConfig>& settings) const override;
};

class RivermaxQueueToIPOReceiverSettingsBuilder
    : public ConversionSettingsBuilder<RivermaxIPOReceiverQueueConfig, IPOReceiverSettings> {
 public:
  RivermaxQueueToIPOReceiverSettingsBuilder(
      std::shared_ptr<RivermaxIPOReceiverQueueConfig> source_settings,
      std::shared_ptr<ISettingsValidator<IPOReceiverSettings>> validator)
      : ConversionSettingsBuilder<RivermaxIPOReceiverQueueConfig, IPOReceiverSettings>(
            source_settings, validator) {}

 protected:
  ReturnStatus convert_settings(
      const std::shared_ptr<RivermaxIPOReceiverQueueConfig>& source_settings,
      std::shared_ptr<IPOReceiverSettings>& target_settings) override;

 public:
  static constexpr int USECS_IN_SECOND = 1000000;
  bool send_packet_ext_info_ = false;

  // Burst pool adaptive dropping configuration
  bool burst_pool_adaptive_dropping_enabled_ = false;
  uint32_t burst_pool_low_threshold_percent_ = 25;
  uint32_t burst_pool_critical_threshold_percent_ = 10;
  uint32_t burst_pool_recovery_threshold_percent_ = 50;

  IPOReceiverSettings built_settings_;
  bool settings_built_ = false;
};

class RivermaxQueueToRTPReceiverSettingsBuilder
    : public ConversionSettingsBuilder<RivermaxRTPReceiverQueueConfig, RTPReceiverSettings> {
 public:
  RivermaxQueueToRTPReceiverSettingsBuilder(
      std::shared_ptr<RivermaxRTPReceiverQueueConfig> source_settings,
      std::shared_ptr<ISettingsValidator<RTPReceiverSettings>> validator)
      : ConversionSettingsBuilder<RivermaxRTPReceiverQueueConfig, RTPReceiverSettings>(
            source_settings, validator) {}

 protected:
  ReturnStatus convert_settings(
      const std::shared_ptr<RivermaxRTPReceiverQueueConfig>& source_settings,
      std::shared_ptr<RTPReceiverSettings>& target_settings) override;

 public:
  static constexpr int USECS_IN_SECOND = 1000000;
  bool send_packet_ext_info_ = false;
  size_t max_chunk_size_ = 0;

  // Burst pool adaptive dropping configuration
  bool burst_pool_adaptive_dropping_enabled_ = false;
  uint32_t burst_pool_low_threshold_percent_ = 25;
  uint32_t burst_pool_critical_threshold_percent_ = 10;
  uint32_t burst_pool_recovery_threshold_percent_ = 50;

  RTPReceiverSettings built_settings_;
  bool settings_built_ = false;
};

class RivermaxQueueToMediaSenderSettingsBuilder
    : public ConversionSettingsBuilder<RivermaxMediaSenderQueueConfig, MediaSenderSettings> {
 public:
  RivermaxQueueToMediaSenderSettingsBuilder(
      std::shared_ptr<RivermaxMediaSenderQueueConfig> source_settings,
      std::shared_ptr<ISettingsValidator<MediaSenderSettings>> validator)
      : ConversionSettingsBuilder<RivermaxMediaSenderQueueConfig, MediaSenderSettings>(
            source_settings, validator) {}

 protected:
  ReturnStatus convert_settings(
      const std::shared_ptr<RivermaxMediaSenderQueueConfig>& source_settings,
      std::shared_ptr<MediaSenderSettings>& target_settings) override;

 public:
  bool dummy_sender_ = false;
  bool use_internal_memory_pool_ = false;
  MemoryKind memory_pool_location_ = MemoryKind::DEVICE;
  MediaSenderSettings built_settings_;
  bool settings_built_ = false;
};

}  // namespace holoscan::advanced_network

#endif  // RIVERMAX_QUEUE_CONFIGS_H_
