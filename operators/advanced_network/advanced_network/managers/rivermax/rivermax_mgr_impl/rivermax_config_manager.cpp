/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/logger/logger.hpp>

#include "rivermax_mgr_impl/burst_manager.h"
#include "rivermax_queue_configs.h"
#include "rivermax_config_manager.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps::rmax_ipo_receiver;
using namespace rivermax::dev_kit::apps::rmax_rtp_receiver;
using namespace rivermax::dev_kit::apps::rmax_xstream_media_sender;

static constexpr int USECS_IN_SECOND = 1000000;

/**
 * @brief Factory class for creating configuration managers.
 *
 * The ConfigManagerFactory class provides a static method to create instances of
 * configuration managers based on the specified configuration type
 */
class ConfigManagerFactory {
 public:
  /**
   * @brief Creates a configuration manager.
   *
   * This static method creates and returns a shared pointer to a configuration manager
   * based on the specified configuration type.
   *
   * @param type The type of configuration manager to create
   * @return A shared pointer to the created configuration manager, or nullptr if the type is
   * invalid.
   */
  static std::shared_ptr<IConfigManager> create_manager(RivermaxConfigContainer::ConfigType type) {
    switch (type) {
      case RivermaxConfigContainer::ConfigType::RX:
        return std::make_shared<RxConfigManager>();
      case RivermaxConfigContainer::ConfigType::TX:
        return std::make_shared<TxConfigManager>();
      default:
        return nullptr;
    }
  }
};

void RivermaxConfigContainer::add_config_manager(ConfigType type,
                                                 std::shared_ptr<IConfigManager> config_manager) {
  config_managers_[type] = std::move(config_manager);
}

void RivermaxConfigContainer::initialize_managers() {
  add_config_manager(ConfigType::RX, ConfigManagerFactory::create_manager(ConfigType::RX));
  add_config_manager(ConfigType::TX, ConfigManagerFactory::create_manager(ConfigType::TX));
}

bool RivermaxConfigContainer::parse_configuration(const NetworkConfig& cfg) {
  int rivermax_rx_config_found = 0;
  int rivermax_tx_config_found = 0;

  is_configured_ = false;
  cfg_ = cfg;

  for (const auto& intf : cfg.ifs_) {
    HOLOSCAN_LOG_INFO("Rivermax init Port {} -- RX: {} TX: {}",
                      intf.port_id_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");

    rivermax_rx_config_found += parse_rx_queues(intf.port_id_, intf.rx_.queues_);
    rivermax_tx_config_found += parse_tx_queues(intf.port_id_, intf.tx_.queues_);
  }

  set_rivermax_log_level(cfg.log_level_);

  if (rivermax_rx_config_found == 0 && rivermax_tx_config_found == 0) {
    HOLOSCAN_LOG_ERROR(
        "Failed to parse Rivermax advanced_network settings. "
        "No valid settings found");
    return false;
  }

  HOLOSCAN_LOG_INFO(
      "Rivermax advanced_network settings were successfully parsed, "
      "Found {} RX Queues and {} TX Queues "
      "settings",
      rivermax_rx_config_found,
      rivermax_tx_config_found);

  is_configured_ = true;
  return true;
}

int RivermaxConfigContainer::parse_rx_queues(uint16_t port_id,
                                             const std::vector<RxQueueConfig>& queues) {
  int rivermax_rx_config_found = 0;

  auto rx_config_manager = std::dynamic_pointer_cast<RxConfigManager>(
      get_config_manager(RivermaxConfigContainer::ConfigType::RX));

  if (!rx_config_manager) {
    return 0;
  }

  rx_config_manager->set_configuration(cfg_);

  for (const auto& q : queues) {
    if (!rx_config_manager->append_candidate_for_rx_queue(port_id, q)) {
      continue;
    }
    rivermax_rx_config_found++;
  }

  return rivermax_rx_config_found;
}

bool RxConfigManager::append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) {
  const auto& queue_id = q.common_.id_;
  uint32_t key = RivermaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  if (config_builder_container_.has_config(key)) {
    HOLOSCAN_LOG_ERROR("Rivermax RX ANO settings for: {} ({}) on port {} already exists",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  HOLOSCAN_LOG_INFO("Configuring RX queue: {} ({}) on port {}", q.common_.name_, queue_id, port_id);

  if (is_configuration_set_ == false) {
    HOLOSCAN_LOG_ERROR("Configuration wasn't set for RxConfigManger");
    return false;
  }

  // extra queue config_ contains Rivermax configuration. If it is not set, return false
  if (!q.common_.extra_queue_config_) {
    HOLOSCAN_LOG_ERROR("Extra queue config is not set for RX queue: {} ({}) on port {}",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  auto* base_rx_config_ptr = dynamic_cast<BaseQueueConfig*>(q.common_.extra_queue_config_);
  if (!base_rx_config_ptr) {
    HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to BaseQueueConfig");
    return false;
  }

  bool res = false;
  auto config_type = base_rx_config_ptr->get_type();
  if (config_type == QueueConfigType::IPOReceiver) {
    auto* rivermax_rx_config_ptr =
        dynamic_cast<RivermaxIPOReceiverQueueConfig*>(q.common_.extra_queue_config_);
    if (!rivermax_rx_config_ptr) {
      HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RivermaxIPOReceiverQueueConfig");
      return false;
    }

    RivermaxIPOReceiverQueueConfig rivermax_rx_config(*rivermax_rx_config_ptr);

    res = append_ipo_receiver_candidate_for_rx_queue(key, q, rivermax_rx_config);
  } else if (config_type == QueueConfigType::RTPReceiver) {
    auto* rivermax_rx_config_ptr =
        dynamic_cast<RivermaxRTPReceiverQueueConfig*>(q.common_.extra_queue_config_);
    if (!rivermax_rx_config_ptr) {
      HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RivermaxRTPReceiverQueueConfig");
      return false;
    }

    RivermaxRTPReceiverQueueConfig rivermax_rx_config(*rivermax_rx_config_ptr);

    res = append_rtp_receiver_candidate_for_rx_queue(key, q, rivermax_rx_config);
  } else {
    HOLOSCAN_LOG_ERROR("Invalid configuration type for Rivermax RX queue: {}",
                       queue_config_type_to_string(config_type));
    return false;
  }

  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to append candidate for RX queue: {} ({}) on port {}",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  HOLOSCAN_LOG_INFO(
      "Rivermax RX ANO settings for {} ({}) on port {}", q.common_.name_, queue_id, port_id);

  return true;
}

bool RxConfigManager::append_ipo_receiver_candidate_for_rx_queue(
    uint32_t config_index, const RxQueueConfig& q,
    RivermaxIPOReceiverQueueConfig& rivermax_rx_config) {
  if (!ConfigManagerUtilities::validate_memory_regions_config(q.common_.mrs_, cfg_.mrs_)) {
    return false;
  }

  if (config_memory_allocator(rivermax_rx_config, q) == false) {
    return false;
  }

  rivermax_rx_config.cpu_cores = q.common_.cpu_core_;
  rivermax_rx_config.master_core = cfg_.common_.master_core_;

  rivermax_rx_config.dump_parameters();
  RivermaxIPOReceiverQueueValidator rivermax_ano_settings_validator;
  auto rivermax_rx_config_ptr =
      std::make_shared<RivermaxIPOReceiverQueueConfig>(rivermax_rx_config);
  auto rc = rivermax_ano_settings_validator.validate(rivermax_rx_config_ptr);
  if (rc != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to validate source settings");
    return false;
  }
  auto rivermax_ipo_receiver_settings_validator = std::make_shared<IPOReceiverSettingsValidator>();
  auto settings_builder = std::make_shared<RivermaxQueueToIPOReceiverSettingsBuilder>(
      std::move(rivermax_rx_config_ptr), std::move(rivermax_ipo_receiver_settings_validator));

  config_builder_container_.add_config_builder(
      config_index, QueueConfigType::IPOReceiver, settings_builder);

  return true;
}

bool RxConfigManager::append_rtp_receiver_candidate_for_rx_queue(
    uint32_t config_index, const RxQueueConfig& q,
    RivermaxRTPReceiverQueueConfig& rivermax_rx_config) {
  if (!ConfigManagerUtilities::validate_memory_regions_config(q.common_.mrs_, cfg_.mrs_)) {
    return false;
  }

  if (config_memory_allocator(rivermax_rx_config, q) == false) {
    return false;
  }

  rivermax_rx_config.cpu_cores = q.common_.cpu_core_;
  rivermax_rx_config.master_core = cfg_.common_.master_core_;

  rivermax_rx_config.dump_parameters();
  RivermaxRTPReceiverQueueValidator rivermax_ano_settings_validator;
  auto rivermax_rx_config_ptr =
      std::make_shared<RivermaxRTPReceiverQueueConfig>(rivermax_rx_config);
  auto rc = rivermax_ano_settings_validator.validate(rivermax_rx_config_ptr);
  if (rc != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to validate source settings");
    return false;
  }
  auto rivermax_rtp_receiver_settings_validator = std::make_shared<RTPReceiverSettingsValidator>();
  auto settings_builder = std::make_shared<RivermaxQueueToRTPReceiverSettingsBuilder>(
      std::move(rivermax_rx_config_ptr), std::move(rivermax_rtp_receiver_settings_validator));

  config_builder_container_.add_config_builder(
      config_index, QueueConfigType::RTPReceiver, settings_builder);

  return true;
}

int RivermaxConfigContainer::parse_tx_queues(uint16_t port_id,
                                             const std::vector<TxQueueConfig>& queues) {
  int rivermax_tx_config_found = 0;

  auto tx_config_manager = std::dynamic_pointer_cast<TxConfigManager>(
      get_config_manager(RivermaxConfigContainer::ConfigType::TX));

  if (!tx_config_manager) {
    return 0;
  }

  tx_config_manager->set_configuration(cfg_);

  for (const auto& q : queues) {
    if (!tx_config_manager->append_candidate_for_tx_queue(port_id, q)) {
      continue;
    }
    rivermax_tx_config_found++;
  }

  return rivermax_tx_config_found;
}

bool TxConfigManager::append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) {
  const auto& queue_id = q.common_.id_;
  uint32_t key = RivermaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  if (config_builder_container_.has_config(key)) {
    HOLOSCAN_LOG_ERROR("Rivermax TX ANO settings for: {} ({}) on port {} already exists",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  HOLOSCAN_LOG_INFO("Configuring TX queue: {} ({}) on port {}", q.common_.name_, queue_id, port_id);

  if (is_configuration_set_ == false) {
    HOLOSCAN_LOG_ERROR("Configuration wasn't set for TxConfigManger");
    return false;
  }

  // extra queue config_ contains Rivermax configuration. If it is not set, return false
  if (!q.common_.extra_queue_config_) {
    HOLOSCAN_LOG_ERROR("Extra queue config is not set for TX queue: {} ({}) on port {}",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  auto* base_tx_config_ptr = dynamic_cast<BaseQueueConfig*>(q.common_.extra_queue_config_);
  if (!base_tx_config_ptr) {
    HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to BaseQueueConfig");
    return false;
  }

  bool res;
  auto config_type = base_tx_config_ptr->get_type();
  if (config_type == QueueConfigType::MediaFrameSender) {
    auto* rivermax_tx_config_ptr =
        dynamic_cast<RivermaxMediaSenderQueueConfig*>(q.common_.extra_queue_config_);
    if (!rivermax_tx_config_ptr) {
      HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RivermaxMediaSenderQueueConfig");
      return false;
    }

    RivermaxMediaSenderQueueConfig rivermax_tx_config(*rivermax_tx_config_ptr);

    res = append_media_sender_candidate_for_tx_queue(key, q, rivermax_tx_config);
  } else {
    HOLOSCAN_LOG_ERROR("Invalid configuration type for Rivermax TX queue: {}",
                       queue_config_type_to_string(config_type));
    return false;
  }

  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to append candidate for TX queue: {} ({}) on port {}",
                       q.common_.name_,
                       queue_id,
                       port_id);
    return false;
  }

  HOLOSCAN_LOG_INFO(
      "Rivermax RX ANO settings for {} ({}) on port {}", q.common_.name_, queue_id, port_id);

  return true;
}

bool TxConfigManager::append_media_sender_candidate_for_tx_queue(
    uint32_t config_index, const TxQueueConfig& q,
    RivermaxMediaSenderQueueConfig& rivermax_tx_config) {
  if (!ConfigManagerUtilities::validate_memory_regions_config(q.common_.mrs_, cfg_.mrs_)) {
    return false;
  }

  if (config_memory_allocator(rivermax_tx_config, q) == false) {
    return false;
  }

  rivermax_tx_config.cpu_cores = q.common_.cpu_core_;
  rivermax_tx_config.master_core = cfg_.common_.master_core_;

  rivermax_tx_config.dump_parameters();

  RivermaxMediaSenderQueueValidator rivermax_ano_settings_validator;
  auto rivermax_tx_config_ptr =
      std::make_shared<RivermaxMediaSenderQueueConfig>(rivermax_tx_config);

  auto rc = rivermax_ano_settings_validator.validate(rivermax_tx_config_ptr);
  if (rc != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to validate source settings");
    return false;
  }

  auto rivermax_media_sender_settings_validator = std::make_shared<MediaSenderSettingsValidator>();
  auto settings_builder = std::make_shared<RivermaxQueueToMediaSenderSettingsBuilder>(
      std::move(rivermax_tx_config_ptr), std::move(rivermax_media_sender_settings_validator));

  settings_builder->dummy_sender_ = rivermax_tx_config.dummy_sender;
  settings_builder->use_internal_memory_pool_ = rivermax_tx_config.use_internal_memory_pool;
  settings_builder->memory_pool_location_ = rivermax_tx_config.memory_pool_location;

  config_builder_container_.add_config_builder(
      config_index, QueueConfigType::MediaFrameSender, settings_builder);

  return true;
}

bool RxConfigManager::config_memory_allocator(RivermaxCommonRxQueueConfig& rivermax_rx_config,
                                              const RxQueueConfig& q) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  HOLOSCAN_LOG_INFO(
      "Configuring memory allocator for Rivermax RX queue: {}, number of memory regions: {}",
      q.common_.name_,
      num_of_mrs);
  if (num_of_mrs == 1) {
    return config_memory_allocator_from_single_mrs(rivermax_rx_config,
                                                   cfg_.mrs_[q.common_.mrs_[0]]);
  } else if (num_of_mrs == 2) {
    return config_memory_allocator_from_dual_mrs(
        rivermax_rx_config, cfg_.mrs_[q.common_.mrs_[0]], cfg_.mrs_[q.common_.mrs_[1]]);
  } else {
    HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax RX queue: {} [1..{}]",
                       num_of_mrs,
                       MAX_RMAX_MEMORY_REGIONS);
    return false;
  }
}

bool RxConfigManager::config_memory_allocator_from_single_mrs(
    RivermaxCommonRxQueueConfig& rivermax_rx_config, const MemoryRegionConfig& mr) {
  rivermax_rx_config.split_boundary = 0;
  rivermax_rx_config.max_packet_size = mr.buf_size_;
  rivermax_rx_config.packets_buffers_size = mr.num_bufs_;

  if (ConfigManagerUtilities::set_gpu_is_in_use_if_applicable(rivermax_rx_config, mr)) {
    return true;
  }

  ConfigManagerUtilities::set_gpu_is_not_in_use(rivermax_rx_config);
  ConfigManagerUtilities::set_cpu_allocator_type(rivermax_rx_config, mr);

  return true;
}

bool RxConfigManager::config_memory_allocator_from_dual_mrs(
    RivermaxCommonRxQueueConfig& rivermax_rx_config, const MemoryRegionConfig& mr_header,
    const MemoryRegionConfig& mr_payload) {
  rivermax_rx_config.split_boundary = mr_header.buf_size_;
  rivermax_rx_config.max_packet_size = mr_payload.buf_size_;
  rivermax_rx_config.packets_buffers_size = mr_payload.num_bufs_;

  if (!ConfigManagerUtilities::set_gpu_is_in_use_if_applicable(rivermax_rx_config, mr_payload)) {
    ConfigManagerUtilities::set_gpu_is_not_in_use(rivermax_rx_config);
  }

  ConfigManagerUtilities::set_cpu_allocator_type(rivermax_rx_config, mr_header);

  return true;
}

bool TxConfigManager::config_memory_allocator(RivermaxCommonTxQueueConfig& rivermax_tx_config,
                                              const TxQueueConfig& q) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  HOLOSCAN_LOG_INFO(
      "Configuring memory allocator for Rivermax TX queue: {}, number of memory regions: {}",
      q.common_.name_,
      num_of_mrs);
  if (num_of_mrs == 1) {
    return config_memory_allocator_from_single_mrs(rivermax_tx_config,
                                                   cfg_.mrs_[q.common_.mrs_[0]]);
  } else if (num_of_mrs == 2) {
    return config_memory_allocator_from_dual_mrs(
        rivermax_tx_config, cfg_.mrs_[q.common_.mrs_[0]], cfg_.mrs_[q.common_.mrs_[1]]);
  } else {
    HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax TX queue: {} [1..{}]",
                       num_of_mrs,
                       MAX_RMAX_MEMORY_REGIONS);
    return false;
  }
}

bool TxConfigManager::config_memory_allocator_from_single_mrs(
    RivermaxCommonTxQueueConfig& rivermax_tx_config, const MemoryRegionConfig& mr) {
  rivermax_tx_config.split_boundary = 0;

  if (ConfigManagerUtilities::set_gpu_is_in_use_if_applicable(rivermax_tx_config, mr)) {
    return true;
  }

  ConfigManagerUtilities::set_gpu_is_not_in_use(rivermax_tx_config);
  ConfigManagerUtilities::set_cpu_allocator_type(rivermax_tx_config, mr);

  return true;
}

bool TxConfigManager::config_memory_allocator_from_dual_mrs(
    RivermaxCommonTxQueueConfig& rivermax_tx_config, const MemoryRegionConfig& mr_header,
    const MemoryRegionConfig& mr_payload) {
  rivermax_tx_config.split_boundary = mr_header.buf_size_;

  if (!ConfigManagerUtilities::set_gpu_is_in_use_if_applicable(rivermax_tx_config, mr_payload)) {
    ConfigManagerUtilities::set_gpu_is_not_in_use(rivermax_tx_config);
  }

  ConfigManagerUtilities::set_cpu_allocator_type(rivermax_tx_config, mr_header);

  return true;
}

Status RivermaxConfigParser::parse_rx_queue_rivermax_config(const YAML::Node& q_item,
                                                            RxQueueConfig& q) {
  const auto& rivermax_rx_settings = q_item["rivermax_rx_settings"];

  if (!rivermax_rx_settings) {
    HOLOSCAN_LOG_ERROR("Rivermax RX settings not found");
    return Status::INVALID_PARAMETER;
  }

  auto settings_type = rivermax_rx_settings["settings_type"].as<std::string>("ipo_receiver");

  if (settings_type == "ipo_receiver") {
    q.common_.extra_queue_config_ = new RivermaxIPOReceiverQueueConfig();
    auto* ipo_rx_config =
        static_cast<RivermaxIPOReceiverQueueConfig*>(q.common_.extra_queue_config_);

    if (!parse_common_rx_settings(rivermax_rx_settings, q_item, *ipo_rx_config)) {
      return Status::INVALID_PARAMETER;
    }

    if (!parse_ipo_receiver_settings(rivermax_rx_settings, *ipo_rx_config)) {
      return Status::INVALID_PARAMETER;
    }
  } else if (settings_type == "rtp_receiver") {
    q.common_.extra_queue_config_ = new RivermaxRTPReceiverQueueConfig();
    auto* rtp_rx_config =
        static_cast<RivermaxRTPReceiverQueueConfig*>(q.common_.extra_queue_config_);

    if (!parse_common_rx_settings(rivermax_rx_settings, q_item, *rtp_rx_config)) {
      return Status::INVALID_PARAMETER;
    }

    if (!parse_rtp_receiver_settings(rivermax_rx_settings, *rtp_rx_config)) {
      return Status::INVALID_PARAMETER;
    }
  } else {
    HOLOSCAN_LOG_ERROR("Invalid settings type for Rivermax RX queue: {}", settings_type);
    return Status::INVALID_PARAMETER;
  }

  return Status::SUCCESS;
}

bool RivermaxConfigParser::parse_common_rx_settings(
    const YAML::Node& rx_settings, const YAML::Node& q_item,
    RivermaxCommonRxQueueConfig& rivermax_rx_config) {
  rivermax_rx_config.ext_seq_num = rx_settings["ext_seq_num"].as<bool>(true);
  rivermax_rx_config.allocator_type = rx_settings["allocator_type"].as<std::string>("auto");
  rivermax_rx_config.memory_registration = rx_settings["memory_registration"].as<bool>(false);
  rivermax_rx_config.sleep_between_operations_us =
      rx_settings["sleep_between_operations_us"].as<int>(0);
  rivermax_rx_config.print_parameters = rx_settings["verbose"].as<bool>(false);
  rivermax_rx_config.num_of_threads = rx_settings["num_of_threads"].as<size_t>(1);
  rivermax_rx_config.send_packet_ext_info = rx_settings["send_packet_ext_info"].as<bool>(true);
  rivermax_rx_config.max_chunk_size = q_item["batch_size"].as<size_t>(1024);
  rivermax_rx_config.stats_report_interval_ms =
      rx_settings["stats_report_interval_ms"].as<uint32_t>(0);

  // Parse burst pool adaptive dropping configuration (optional)
  const auto& burst_pool_config = rx_settings["burst_pool_adaptive_dropping"];
  if (burst_pool_config) {
    rivermax_rx_config.burst_pool_adaptive_dropping_enabled =
        burst_pool_config["enabled"].as<bool>(false);
    rivermax_rx_config.burst_pool_low_threshold_percent =
        burst_pool_config["low_threshold_percent"].as<uint32_t>(25);
    rivermax_rx_config.burst_pool_critical_threshold_percent =
        burst_pool_config["critical_threshold_percent"].as<uint32_t>(10);
    rivermax_rx_config.burst_pool_recovery_threshold_percent =
        burst_pool_config["recovery_threshold_percent"].as<uint32_t>(50);

    // Validate threshold percentages
    uint32_t critical = rivermax_rx_config.burst_pool_critical_threshold_percent;
    uint32_t low = rivermax_rx_config.burst_pool_low_threshold_percent;
    uint32_t recovery = rivermax_rx_config.burst_pool_recovery_threshold_percent;

    // Check valid range (0..100)
    if (critical > 100 || low > 100 || recovery > 100) {
      HOLOSCAN_LOG_ERROR(
          "Invalid burst pool threshold percentages: all values must be in range 0..100 "
          "(critical={}, low={}, recovery={})",
          critical, low, recovery);
      return false;
    }

    // Check for nonsensical zero values
    if (critical == 0 || recovery == 0) {
      HOLOSCAN_LOG_ERROR(
          "Invalid burst pool threshold percentages: critical and recovery cannot be 0 "
          "(critical={}, low={}, recovery={})",
          critical, low, recovery);
      return false;
    }

    // Check proper ordering: critical < low < recovery
    if (critical >= low) {
      HOLOSCAN_LOG_ERROR(
          "Invalid burst pool threshold ordering: critical must be < low "
          "(critical={}, low={})",
          critical, low);
      return false;
    }

    if (low >= recovery) {
      HOLOSCAN_LOG_ERROR(
          "Invalid burst pool threshold ordering: low must be < recovery "
          "(low={}, recovery={})",
          low, recovery);
      return false;
    }

    HOLOSCAN_LOG_INFO(
        "Parsed burst pool adaptive dropping config: enabled={}, thresholds={}%/{}%/{}%",
        rivermax_rx_config.burst_pool_adaptive_dropping_enabled,
        rivermax_rx_config.burst_pool_low_threshold_percent,
        rivermax_rx_config.burst_pool_critical_threshold_percent,
        rivermax_rx_config.burst_pool_recovery_threshold_percent);
  } else {
    // Use default values if not specified
    rivermax_rx_config.burst_pool_adaptive_dropping_enabled = false;
    rivermax_rx_config.burst_pool_low_threshold_percent = 25;
    rivermax_rx_config.burst_pool_critical_threshold_percent = 10;
    rivermax_rx_config.burst_pool_recovery_threshold_percent = 50;
  }

  return true;
}

bool RivermaxConfigParser::parse_ipo_receiver_settings(
    const YAML::Node& rx_settings, RivermaxIPOReceiverQueueConfig& rivermax_rx_config) {
  for (const auto& q_item_ip : rx_settings["local_ip_addresses"]) {
    rivermax_rx_config.local_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rx_settings["source_ip_addresses"]) {
    rivermax_rx_config.source_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rx_settings["destination_ip_addresses"]) {
    rivermax_rx_config.destination_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rx_settings["destination_ports"]) {
    rivermax_rx_config.destination_ports.emplace_back(q_item_ip.as<uint16_t>());
  }

  rivermax_rx_config.max_path_differential_us = rx_settings["max_path_diff_us"].as<uint32_t>(0);

  return true;
}

bool RivermaxConfigParser::parse_rtp_receiver_settings(
    const YAML::Node& rx_settings, RivermaxRTPReceiverQueueConfig& rivermax_rx_config) {
  rivermax_rx_config.local_ip = rx_settings["local_ip_address"].as<std::string>("");
  rivermax_rx_config.source_ip = rx_settings["source_ip_address"].as<std::string>("");
  rivermax_rx_config.destination_ip = rx_settings["destination_ip_address"].as<std::string>("");
  rivermax_rx_config.destination_port = rx_settings["destination_port"].as<uint16_t>(0);

  return true;
}

Status RivermaxConfigParser::parse_tx_queue_rivermax_config(const YAML::Node& q_item,
                                                            TxQueueConfig& q) {
  const auto& rivermax_tx_settings = q_item["rivermax_tx_settings"];

  if (!rivermax_tx_settings) {
    HOLOSCAN_LOG_ERROR("Rivermax TX settings not found");
    return Status::INVALID_PARAMETER;
  }

  auto settings_type = rivermax_tx_settings["settings_type"].as<std::string>("media_sender");

  if (settings_type == "media_sender") {
    q.common_.extra_queue_config_ = new RivermaxMediaSenderQueueConfig();
    auto* media_tx_config =
        static_cast<RivermaxMediaSenderQueueConfig*>(q.common_.extra_queue_config_);

    if (!parse_common_tx_settings(rivermax_tx_settings, q_item, *media_tx_config)) {
      return Status::INVALID_PARAMETER;
    }

    if (!parse_media_sender_settings(rivermax_tx_settings, *media_tx_config)) {
      return Status::INVALID_PARAMETER;
    }
  } else if (settings_type == "generic_sender") {
    HOLOSCAN_LOG_ERROR("Generic Sender is not supported");
    return Status::INVALID_PARAMETER;
  } else {
    HOLOSCAN_LOG_ERROR("Invalid settings type for Rivermax TX queue: {}", settings_type);
    return Status::INVALID_PARAMETER;
  }

  return Status::SUCCESS;
}

bool RivermaxConfigParser::parse_common_tx_settings(
    const YAML::Node& tx_settings, const YAML::Node& q_item,
    RivermaxCommonTxQueueConfig& rivermax_tx_config) {
  rivermax_tx_config.local_ip = tx_settings["local_ip_address"].as<std::string>("");
  rivermax_tx_config.destination_ip = tx_settings["destination_ip_address"].as<std::string>("");
  rivermax_tx_config.destination_port = tx_settings["destination_port"].as<uint16_t>(0);

  rivermax_tx_config.allocator_type = tx_settings["allocator_type"].as<std::string>("auto");
  rivermax_tx_config.memory_registration = tx_settings["memory_registration"].as<bool>(true);
  rivermax_tx_config.memory_allocation = tx_settings["memory_allocation"].as<bool>(true);
  rivermax_tx_config.lock_gpu_clocks = tx_settings["lock_gpu_clocks"].as<bool>(true);

  rivermax_tx_config.print_parameters = tx_settings["verbose"].as<bool>(false);
  rivermax_tx_config.num_of_threads = tx_settings["num_of_threads"].as<size_t>(1);
  rivermax_tx_config.send_packet_ext_info = tx_settings["send_packet_ext_info"].as<bool>(true);
  rivermax_tx_config.num_of_packets_in_chunk = tx_settings["num_of_packets_in_chunk"].as<size_t>(
      MediaSenderSettings::DEFAULT_NUM_OF_PACKETS_IN_CHUNK_FHD);
  rivermax_tx_config.sleep_between_operations =
      tx_settings["sleep_between_operations"].as<bool>(true);
  rivermax_tx_config.stats_report_interval_ms =
      tx_settings["stats_report_interval_ms"].as<uint32_t>(1000);
  rivermax_tx_config.dummy_sender = tx_settings["dummy_sender"].as<bool>(false);

  return true;
}

bool RivermaxConfigParser::parse_media_sender_settings(
    const YAML::Node& tx_settings, RivermaxMediaSenderQueueConfig& rivermax_tx_config) {
  rivermax_tx_config.video_format = tx_settings["video_format"].as<std::string>("");
  rivermax_tx_config.bit_depth = tx_settings["bit_depth"].as<uint16_t>(0);
  rivermax_tx_config.frame_width = tx_settings["frame_width"].as<uint16_t>(0);
  rivermax_tx_config.frame_height = tx_settings["frame_height"].as<uint16_t>(0);
  rivermax_tx_config.frame_rate = tx_settings["frame_rate"].as<uint16_t>(0);
  rivermax_tx_config.use_internal_memory_pool =
      tx_settings["use_internal_memory_pool"].as<bool>(false);
  if (rivermax_tx_config.use_internal_memory_pool) {
    rivermax_tx_config.memory_pool_location = GetMemoryKindFromString(
        tx_settings["memory_pool_location"].template as<std::string>("device"));
    if (rivermax_tx_config.memory_pool_location == MemoryKind::INVALID) {
      rivermax_tx_config.memory_pool_location = MemoryKind::DEVICE;
      HOLOSCAN_LOG_ERROR("Invalid memory pool location, setting to DEVICE");
    }
  } else {
    rivermax_tx_config.memory_pool_location = MemoryKind::INVALID;
  }
  return true;
}

template <typename T>
void ConfigManagerUtilities::set_cpu_allocator_type(T& rivermax_config,
                                                    const MemoryRegionConfig& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::HOST) {
#else
  if (mr.kind_ == MemoryKind::HOST || mr.kind_ == MemoryKind::HOST_PINNED) {
#endif
    rivermax_config.allocator_type = "malloc";
  } else if (mr.kind_ == MemoryKind::HUGE) {
    if (rivermax_config.allocator_type != "huge_page_default" &&
        rivermax_config.allocator_type != "huge_page_2mb" &&
        rivermax_config.allocator_type != "huge_page_512mb" &&
        rivermax_config.allocator_type != "huge_page_1gb") {
      rivermax_config.allocator_type = "huge_page_default";
    }  // else the allocator type is already set
  }
}

template <typename T>
void ConfigManagerUtilities::set_gpu_is_not_in_use(T& rivermax_config) {
  rivermax_config.gpu_device_id = -1;
  rivermax_config.gpu_direct = false;
}

template <typename T>
bool ConfigManagerUtilities::set_gpu_is_in_use_if_applicable(T& rivermax_config,
                                                             const MemoryRegionConfig& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::DEVICE || mr.kind_ == MemoryKind::HOST_PINNED) {
#else
  if (mr.kind_ == MemoryKind::DEVICE) {
#endif
    rivermax_config.gpu_device_id = mr.affinity_;
    rivermax_config.gpu_direct = true;
    return true;
  }
  return false;
}

bool ConfigManagerUtilities::parse_and_set_cores(std::vector<int>& app_threads_cores,
                                                 const std::string& cores) {
  std::istringstream iss(cores);
  std::string coreStr;
  bool to_reset_cores_vector = true;
  while (std::getline(iss, coreStr, ',')) {
    try {
      int core = std::stoi(coreStr);
      if (core < 0 || core >= std::thread::hardware_concurrency()) {
        HOLOSCAN_LOG_ERROR("Invalid core number: {}", coreStr);
        return false;
      } else {
        if (to_reset_cores_vector) {
          app_threads_cores.clear();
          to_reset_cores_vector = false;
        }
        app_threads_cores.push_back(core);
      }
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid core number: {}", coreStr);
      return false;
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Core number out of range: {}", coreStr);
      return false;
    }
  }
  return true;
}

bool ConfigManagerUtilities::validate_cores(const std::string& cores) {
  std::istringstream iss(cores);
  std::string coreStr;
  bool to_reset_cores_vector = true;
  while (std::getline(iss, coreStr, ',')) {
    try {
      int core = std::stoi(coreStr);
      if (core < 0 || core >= std::thread::hardware_concurrency()) {
        HOLOSCAN_LOG_ERROR("Invalid core number: {}", coreStr);
        return false;
      }
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid core number: {}", coreStr);
      return false;
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Core number out of range: {}", coreStr);
      return false;
    }
  }
  return true;
}

void ConfigManagerUtilities::set_allocator_type(AppSettings& app_settings_config,
                                                const std::string& allocator_type) {
  auto setAllocatorType = [&](const std::string& allocatorTypeStr, AllocatorTypeUI allocatorType) {
    if (allocator_type == allocatorTypeStr) {
      app_settings_config.allocator_type = allocatorType;
    }
  };

  app_settings_config.allocator_type = AllocatorTypeUI::Auto;
  setAllocatorType("auto", AllocatorTypeUI::Auto);
  setAllocatorType("huge_page_default", AllocatorTypeUI::HugePageDefault);
  setAllocatorType("malloc", AllocatorTypeUI::Malloc);
  setAllocatorType("huge_page_2mb", AllocatorTypeUI::HugePage2MB);
  setAllocatorType("huge_page_512mb", AllocatorTypeUI::HugePage512MB);
  setAllocatorType("huge_page_1gb", AllocatorTypeUI::HugePage1GB);
  setAllocatorType("gpu", AllocatorTypeUI::GPU);
}

VideoSampling ConfigManagerUtilities::convert_video_sampling(const std::string& sampling) {
  static const std::map<std::string, VideoSampling> videoSamplingMap = {
      {"YCbCr-4:4:4", VideoSampling::YCbCr_4_4_4},
      {"YCbCr-4:2:2", VideoSampling::YCbCr_4_2_2},
      {"YCbCr-4:2:0", VideoSampling::YCbCr_4_2_0},
      {"CLYCbCr-4:4:4", VideoSampling::CLYCbCr_4_4_4},
      {"CLYCbCr-4:2:2", VideoSampling::CLYCbCr_4_2_2},
      {"CLYCbCr-4:2:0", VideoSampling::CLYCbCr_4_2_0},
      {"ICtCp-4:4:4", VideoSampling::ICtCp_4_4_4},
      {"ICtCp-4:2:2", VideoSampling::ICtCp_4_2_2},
      {"ICtCp-4:2:0", VideoSampling::ICtCp_4_2_0},
      {"RGB", VideoSampling::RGB},
      {"XYZ", VideoSampling::XYZ},
      {"KEY", VideoSampling::KEY}};

  auto it = videoSamplingMap.find(sampling);
  if (it != videoSamplingMap.end()) {
    return it->second;
  } else {
    return VideoSampling::Unknown;
  }
}

ColorBitDepth ConfigManagerUtilities::convert_bit_depth(uint16_t bit_depth) {
  static const std::map<uint16_t, ColorBitDepth> bitDepthMap = {{8, ColorBitDepth::_8},
                                                                {10, ColorBitDepth::_10},
                                                                {12, ColorBitDepth::_12},
                                                                {16, ColorBitDepth::_16},
                                                                {165, ColorBitDepth::_16f}};

  auto it = bitDepthMap.find(bit_depth);
  if (it != bitDepthMap.end()) {
    return it->second;
  } else {
    return ColorBitDepth::Unknown;
  }
}

bool ConfigManagerUtilities::validate_memory_regions_config(
    const std::vector<std::string>& queue_mr_names,
    const std::unordered_map<std::string, MemoryRegionConfig>& memory_regions) {
  uint16_t num_of_mrs = queue_mr_names.size();
  try {
    if (num_of_mrs == 1) {
      return validate_memory_regions_config_from_single_mrs(memory_regions.at(queue_mr_names[0]));
    } else if (num_of_mrs == 2) {
      return validate_memory_regions_config_from_dual_mrs(memory_regions.at(queue_mr_names[0]),
                                                          memory_regions.at(queue_mr_names[1]));
    } else {
      HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax RX queue: {} [1..2]",
                         num_of_mrs);
      return false;
    }
  } catch (const std::out_of_range& e) {
    if (num_of_mrs == 1)
      HOLOSCAN_LOG_ERROR("Invalid memory region for Rivermax RX queue: {}", queue_mr_names[0]);
    else
      HOLOSCAN_LOG_ERROR("Invalid memory region for Rivermax RX queue: {} or {}",
                         queue_mr_names[0],
                         queue_mr_names[1]);
    return false;
  }

  return true;
}

bool ConfigManagerUtilities::validate_memory_regions_config_from_single_mrs(
    const MemoryRegionConfig& mr) {
  return true;
}

bool ConfigManagerUtilities::validate_memory_regions_config_from_dual_mrs(
    const MemoryRegionConfig& mr_header, const MemoryRegionConfig& mr_payload) {
  if (mr_payload.kind_ != MemoryKind::DEVICE && mr_header.kind_ != mr_payload.kind_) {
    HOLOSCAN_LOG_ERROR(
        "Memory region kind mismatch: {} != {}", (int)(mr_header.kind_), (int)mr_payload.kind_);
    return false;
  }

  if (mr_payload.kind_ == MemoryKind::DEVICE && mr_header.kind_ == MemoryKind::DEVICE) {
    HOLOSCAN_LOG_ERROR("Both memory regions are device memory");
    return false;
  }

  if (mr_payload.buf_size_ == 0) {
    HOLOSCAN_LOG_ERROR("Invalid payload memory region size: {}", mr_payload.buf_size_);
    return false;
  }

  return true;
}

}  // namespace holoscan::advanced_network
