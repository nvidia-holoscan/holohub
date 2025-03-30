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

#include "rt_threads.h"
#include "rmax_ipo_receiver_service.h"
#include "rivermax_mgr_impl/burst_manager.h"
#include <holoscan/logger/logger.hpp>

#include "rivermax_config_manager.h"

namespace holoscan::advanced_network {

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
  config_managers_[type] = config_manager;
}

void RivermaxConfigContainer::initialize_managers() {
  add_config_manager(ConfigType::RX, ConfigManagerFactory::create_manager(ConfigType::RX));
  add_config_manager(ConfigType::TX, ConfigManagerFactory::create_manager(ConfigType::TX));
}

int RivermaxConfigContainer::parse_rx_queues(uint16_t port_id,
                                         const std::vector<RxQueueConfig>& queues) {
  int rivermax_rx_config_found = 0;

  auto rx_manager = std::dynamic_pointer_cast<RxConfigManager>(
      get_config_manager(RivermaxConfigContainer::ConfigType::RX));

  if (!rx_manager) { return 0; }

  rx_manager->set_configuration(cfg_, rmax_apps_lib_);

  for (const auto& q : queues) {
    if (!rx_manager->append_candidate_for_rx_queue(port_id, q)) { continue; }
    rivermax_rx_config_found++;
  }

  return rivermax_rx_config_found;
}

int RivermaxConfigContainer::parse_tx_queues(uint16_t port_id,
                                         const std::vector<TxQueueConfig>& queues) {
  int rivermax_tx_config_found = 0;

  auto tx_manager = std::dynamic_pointer_cast<TxConfigManager>(
      get_config_manager(RivermaxConfigContainer::ConfigType::TX));

  if (!tx_manager) { return 0; }

  tx_manager->set_configuration(cfg_, rmax_apps_lib_);

  for (const auto& q : queues) {
    if (!tx_manager->append_candidate_for_tx_queue(port_id, q)) { continue; }
    rivermax_tx_config_found++;
  }

  return rivermax_tx_config_found;
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
    HOLOSCAN_LOG_ERROR("Failed to parse Rivermax advanced_network settings. "
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

void RxConfigManager::set_default_config(ExtRmaxIPOReceiverConfig& rx_service_cfg) const {
  rx_service_cfg.app_settings->destination_ip = DESTINATION_IP_DEFAULT;
  rx_service_cfg.app_settings->destination_port = DESTINATION_PORT_DEFAULT;
  rx_service_cfg.app_settings->num_of_threads = NUM_OF_THREADS_DEFAULT;
  rx_service_cfg.app_settings->num_of_total_streams = NUM_OF_TOTAL_STREAMS_DEFAULT;
  rx_service_cfg.app_settings->num_of_total_flows = NUM_OF_TOTAL_FLOWS_DEFAULT;
  rx_service_cfg.app_settings->internal_thread_core = CPU_NONE;
  rx_service_cfg.app_settings->app_threads_cores =
      std::vector<int>(rx_service_cfg.app_settings->num_of_threads, CPU_NONE);
  rx_service_cfg.app_settings->rate = {0, 0};
  rx_service_cfg.app_settings->num_of_chunks = NUM_OF_CHUNKS_DEFAULT;
  rx_service_cfg.app_settings->num_of_packets_in_chunk = NUM_OF_PACKETS_IN_CHUNK_DEFAULT;
  rx_service_cfg.app_settings->packet_payload_size = PACKET_PAYLOAD_SIZE_DEFAULT;
  rx_service_cfg.app_settings->packet_app_header_size = PACKET_APP_HEADER_SIZE_DEFAULT;
  rx_service_cfg.app_settings->sleep_between_operations_us = SLEEP_BETWEEN_OPERATIONS_US_DEFAULT;
  rx_service_cfg.app_settings->sleep_between_operations = false;
  rx_service_cfg.app_settings->print_parameters = false;
  rx_service_cfg.app_settings->use_checksum_header = false;
  rx_service_cfg.app_settings->hw_queue_full_sleep_us = 0;
  rx_service_cfg.app_settings->gpu_id = INVALID_GPU_ID;
  rx_service_cfg.app_settings->allocator_type = AllocatorTypeUI::Auto;
  rx_service_cfg.app_settings->statistics_reader_core = INVALID_CORE_NUMBER;
  rx_service_cfg.app_settings->session_id_stats = UINT_MAX;
  rx_service_cfg.is_extended_sequence_number = true;
  rx_service_cfg.max_path_differential_us = 0;
  rx_service_cfg.register_memory = false;
  rx_service_cfg.max_chunk_size = 0;
  rx_service_cfg.rmax_apps_lib = nullptr;
  rx_service_cfg.rx_stats_period_report_ms = 1000;
}

bool RxConfigManager::append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) {
  HOLOSCAN_LOG_INFO(
      "Configuring RX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, port_id);

  if (is_configuration_set_ == false) {
    HOLOSCAN_LOG_ERROR("Configuration wasn't set for RxConfigManger");
    return false;
  }

  // extra queue config_ contains Rivermax configuration. If it is not set, return false
  if (!q.common_.extra_queue_config_) return false;

  auto* rivermax_rx_config_ptr = dynamic_cast<RivermaxRxQueueConfig*>(q.common_.extra_queue_config_);
  if (!rivermax_rx_config_ptr) {
    HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RivermaxRxQueueConfig");
    return false;
  }

  RivermaxRxQueueConfig rivermax_rx_config(*rivermax_rx_config_ptr);

  if (!validate_rx_queue_config(rivermax_rx_config)) { return false; }

  if (!validate_memory_regions_config(q, rivermax_rx_config)) { return false; }

  if (config_memory_allocator(rivermax_rx_config, q) == false) { return false; }

  rivermax_rx_config.dump_parameters();

  ExtRmaxIPOReceiverConfig rx_service_cfg;

  if (!build_rmax_ipo_receiver_config(rx_service_cfg, rivermax_rx_config, q)) { return false; }

  add_new_rx_service_config(rx_service_cfg, port_id, q.common_.id_);

  return true;
}

bool RxConfigManager::config_memory_allocator(RivermaxRxQueueConfig& rivermax_rx_config,
                                              const RxQueueConfig& q) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  HOLOSCAN_LOG_INFO(
      "Configuring memory allocator for Rivermax RX queue: {}, number of memory regions: {}",
      q.common_.name_,
      num_of_mrs);
  if (num_of_mrs == 1) {
    return config_memory_allocator_from_single_mrs(rivermax_rx_config, q, cfg_.mrs_[q.common_.mrs_[0]]);
  } else if (num_of_mrs == 2) {
    return config_memory_allocator_from_dual_mrs(
        rivermax_rx_config, q, cfg_.mrs_[q.common_.mrs_[0]], cfg_.mrs_[q.common_.mrs_[1]]);
  } else {
    HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax RX queue: {} [1..{}]",
                       num_of_mrs,
                       MAX_RMAX_MEMORY_REGIONS);
    return false;
  }
}

bool RxConfigManager::config_memory_allocator_from_single_mrs(RivermaxRxQueueConfig& rivermax_rx_config,
                                                              const RxQueueConfig& q,
                                                              const MemoryRegionConfig& mr) {
  rivermax_rx_config.split_boundary = 0;
  rivermax_rx_config.max_packet_size = mr.buf_size_;
  rivermax_rx_config.packets_buffers_size = mr.num_bufs_;

  if (set_gpu_is_in_use_if_applicable(rivermax_rx_config, mr)) { return true; }

  set_gpu_is_not_in_use(rivermax_rx_config);
  set_cpu_allocator_type(rivermax_rx_config, mr);

  return true;
}

bool RxConfigManager::config_memory_allocator_from_dual_mrs(RivermaxRxQueueConfig& rivermax_rx_config,
                                                            const RxQueueConfig& q,
                                                            const MemoryRegionConfig& mr_header,
                                                            const MemoryRegionConfig& mr_payload) {
  rivermax_rx_config.split_boundary = mr_header.buf_size_;
  rivermax_rx_config.max_packet_size = mr_payload.buf_size_;
  rivermax_rx_config.packets_buffers_size = mr_payload.num_bufs_;

  if (!set_gpu_is_in_use_if_applicable(rivermax_rx_config, mr_payload)) {
    set_gpu_is_not_in_use(rivermax_rx_config);
  }

  set_cpu_allocator_type(rivermax_rx_config, mr_header);

  return true;
}

bool RxConfigManager::set_gpu_is_in_use_if_applicable(RivermaxRxQueueConfig& rivermax_rx_config,
                                                      const MemoryRegionConfig& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::DEVICE || mr.kind_ == MemoryKind::HOST_PINNED) {
#else
  if (mr.kind_ == MemoryKind::DEVICE) {
#endif
    rivermax_rx_config.gpu_device_id = mr.affinity_;
    rivermax_rx_config.gpu_direct = true;
    return true;
  }
  return false;
}

void RxConfigManager::set_gpu_is_not_in_use(RivermaxRxQueueConfig& rivermax_rx_config) {
  rivermax_rx_config.gpu_device_id = -1;
  rivermax_rx_config.gpu_direct = false;
}

void RxConfigManager::set_cpu_allocator_type(RivermaxRxQueueConfig& rivermax_rx_config,
                                             const MemoryRegionConfig& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::HOST) {
#else
  if (mr.kind_ == MemoryKind::HOST || mr.kind_ == MemoryKind::HOST_PINNED) {
#endif

    rivermax_rx_config.allocator_type = "malloc";
  } else if (mr.kind_ == MemoryKind::HUGE) {
    if (rivermax_rx_config.allocator_type != "huge_page_default" &&
        rivermax_rx_config.allocator_type != "huge_page_2mb" &&
        rivermax_rx_config.allocator_type != "huge_page_512mb" &&
        rivermax_rx_config.allocator_type != "huge_page_1gb") {
      rivermax_rx_config.allocator_type = "huge_page_default";
    }  // else the allocator type is already set
  }
}

bool RxConfigManager::validate_memory_regions_config(const RxQueueConfig& q,
                                                     const RivermaxRxQueueConfig& rivermax_rx_config) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  try {
    if (num_of_mrs == 1) {
      return validate_memory_regions_config_from_single_mrs(
          q, rivermax_rx_config, cfg_.mrs_.at(q.common_.mrs_[0]));
    } else if (num_of_mrs == 2) {
      return validate_memory_regions_config_from_dual_mrs(
          q, rivermax_rx_config, cfg_.mrs_.at(q.common_.mrs_[0]), cfg_.mrs_.at(q.common_.mrs_[1]));
    } else {
      HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax RX queue: {} [1..{}]",
                         num_of_mrs,
                         MAX_RMAX_MEMORY_REGIONS);
      return false;
    }
  } catch (const std::out_of_range& e) {
    if (num_of_mrs == 1)
      HOLOSCAN_LOG_ERROR("Invalid memory region for Rivermax RX queue: {}", q.common_.mrs_[0]);
    else
      HOLOSCAN_LOG_ERROR("Invalid memory region for Rivermax RX queue: {} or {}",
                         q.common_.mrs_[0],
                         q.common_.mrs_[1]);
    return false;
  }

  return true;
}

bool RxConfigManager::validate_memory_regions_config_from_single_mrs(
    const RxQueueConfig& q, const RivermaxRxQueueConfig& rivermax_rx_config, const MemoryRegionConfig& mr) {
  return true;
}

/**
 * @brief Validates the RX queue memory regions configuration for dual memory regions.
 *
 * @param q The RX queue configuration.
 * @param rivermax_rx_config The Rivermax RX queue configuration.
 * @param mr_header The header memory region.
 * @param mr_payload The payload memory region.
 * @return True if the configuration is valid, false otherwise.
 */
bool RxConfigManager::validate_memory_regions_config_from_dual_mrs(
    const RxQueueConfig& q, const RivermaxRxQueueConfig& rivermax_rx_config,
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

bool RxConfigManager::build_rmax_ipo_receiver_config(ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                                     const RivermaxRxQueueConfig& rivermax_rx_config,
                                                     const RxQueueConfig& q) {
  rx_service_cfg.app_settings = std::make_shared<AppSettings>();
  set_default_config(rx_service_cfg);

  auto& app_settings_config = *(rx_service_cfg.app_settings);

  set_rx_service_common_app_settings(app_settings_config, rivermax_rx_config);

  if (!parse_and_set_cores(app_settings_config, q.common_.cpu_core_)) { return false; }

  set_rx_service_ipo_receiver_settings(rx_service_cfg, rivermax_rx_config);

  return true;
}

bool RxConfigManager::validate_rx_queue_config(const RivermaxRxQueueConfig& rivermax_rx_config) {
  if (rivermax_rx_config.source_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Source IP addresses are not set for RTP stream");
    return false;
  }

  if (rivermax_rx_config.local_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Local IP addresses are not set for RTP stream");
    return false;
  }

  if (rivermax_rx_config.destination_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Destination IP addresses are not set for RTP stream");
    return false;
  }

  if (rivermax_rx_config.destination_ports.empty()) {
    HOLOSCAN_LOG_ERROR("Destination ports are not set for RTP stream");
    return false;
  }

  if ((rivermax_rx_config.local_ips.size() != rivermax_rx_config.source_ips.size()) ||
      (rivermax_rx_config.local_ips.size() != rivermax_rx_config.destination_ips.size()) ||
      (rivermax_rx_config.local_ips.size() != rivermax_rx_config.destination_ports.size())) {
    HOLOSCAN_LOG_ERROR(
        "Local/Source/Destination IP addresses and ports sizes are not equal for RTP stream");
    return false;
  }

  return true;
}

void RxConfigManager::set_rx_service_common_app_settings(AppSettings& app_settings_config,
                                                         const RivermaxRxQueueConfig& rivermax_rx_config) {
  app_settings_config.local_ips = rivermax_rx_config.local_ips;
  app_settings_config.source_ips = rivermax_rx_config.source_ips;
  app_settings_config.destination_ips = rivermax_rx_config.destination_ips;
  app_settings_config.destination_ports = rivermax_rx_config.destination_ports;

  if (rivermax_rx_config.gpu_direct) {
    app_settings_config.gpu_id = rivermax_rx_config.gpu_device_id;
  } else {
    app_settings_config.gpu_id = INVALID_GPU_ID;
  }

  set_allocator_type(app_settings_config, rivermax_rx_config.allocator_type);

  if (cfg_.common_.master_core_ >= 0 &&
      cfg_.common_.master_core_ < std::thread::hardware_concurrency()) {
    app_settings_config.internal_thread_core = cfg_.common_.master_core_;
  } else {
    app_settings_config.internal_thread_core = CPU_NONE;
  }
  app_settings_config.num_of_threads = rivermax_rx_config.num_of_threads;
  app_settings_config.print_parameters = rivermax_rx_config.print_parameters;
  app_settings_config.sleep_between_operations_us = rivermax_rx_config.sleep_between_operations_us;
  app_settings_config.packet_payload_size = rivermax_rx_config.max_packet_size;
  app_settings_config.packet_app_header_size = rivermax_rx_config.split_boundary;
  app_settings_config.num_of_packets_in_chunk =
      std::pow(2, std::ceil(std::log2(rivermax_rx_config.packets_buffers_size)));
}

void RxConfigManager::set_allocator_type(AppSettings& app_settings_config,
                                         const std::string& allocator_type) {
  auto setAllocatorType = [&](const std::string& allocatorTypeStr, AllocatorTypeUI allocatorType) {
    if (allocator_type == allocatorTypeStr) { app_settings_config.allocator_type = allocatorType; }
  };

  app_settings_config.allocator_type = AllocatorTypeUI::Auto;
  setAllocatorType("auto", AllocatorTypeUI::Auto);
  setAllocatorType("huge_page_default", AllocatorTypeUI::HugePageDefault);
  setAllocatorType("malloc", AllocatorTypeUI::Malloc);
  setAllocatorType("huge_page_2mb", AllocatorTypeUI::HugePage2MB);
  setAllocatorType("huge_page_512mb", AllocatorTypeUI::HugePage512MB);
  setAllocatorType("huge_page_1gb", AllocatorTypeUI::HugePage1GB);
  setAllocatorType("gpu", AllocatorTypeUI::Gpu);
}

bool RxConfigManager::parse_and_set_cores(AppSettings& app_settings_config,
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
          app_settings_config.app_threads_cores.clear();
          to_reset_cores_vector = false;
        }
        app_settings_config.app_threads_cores.push_back(core);
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

void RxConfigManager::set_rx_service_ipo_receiver_settings(
    ExtRmaxIPOReceiverConfig& rx_service_cfg, const RivermaxRxQueueConfig& rivermax_rx_config) {
  rx_service_cfg.is_extended_sequence_number = rivermax_rx_config.ext_seq_num;
  rx_service_cfg.max_path_differential_us = rivermax_rx_config.max_path_differential_us;
  if (rx_service_cfg.max_path_differential_us >= USECS_IN_SECOND) {
    HOLOSCAN_LOG_ERROR("Max path differential must be less than 1 second");
    rx_service_cfg.max_path_differential_us = USECS_IN_SECOND;
  }

  rx_service_cfg.rx_stats_period_report_ms = rivermax_rx_config.rx_stats_period_report_ms;
  rx_service_cfg.register_memory = rivermax_rx_config.memory_registration;
  rx_service_cfg.max_chunk_size = rivermax_rx_config.max_chunk_size;
  rx_service_cfg.rmax_apps_lib = this->rmax_apps_lib_;

  rx_service_cfg.send_packet_ext_info = rivermax_rx_config.send_packet_ext_info;
}

void RxConfigManager::add_new_rx_service_config(const ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                                uint16_t port_id, uint16_t queue_id) {
  uint32_t key = RivermaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  if (rx_service_configs_.find(key) != rx_service_configs_.end()) {
    HOLOSCAN_LOG_ERROR("Rivermax advanced_network settings for port {} and queue {} already exists",
                       port_id,
                       queue_id);
    return;
  }
  HOLOSCAN_LOG_INFO("Rivermax advanced_network settings for port {} and queue {} added",
                    port_id,
                    queue_id);

  rx_service_configs_[key] = rx_service_cfg;
}

bool TxConfigManager::append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) {
  HOLOSCAN_LOG_INFO(
      "Configuring TX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, port_id);

  if (is_configuration_set_ == false) {
    HOLOSCAN_LOG_ERROR("Configuration wasn't set for TxConfigManger");
    return false;
  }

  // TODO: Implement when TX is ready
  return false;
}

Status RivermaxConfigParser::parse_rx_queue_rivermax_config(const YAML::Node& q_item,
                                                        RxQueueConfig& q) {
  const auto& rivermax_rx_settings = q_item["rivermax_rx_settings"];

  if (!rivermax_rx_settings) {
    HOLOSCAN_LOG_ERROR("Rivermax RX settings not found");
    return Status::INVALID_PARAMETER;
  }

  q.common_.extra_queue_config_ = new RivermaxRxQueueConfig();
  auto& rivermax_rx_config = *(reinterpret_cast<RivermaxRxQueueConfig*>(q.common_.extra_queue_config_));

  for (const auto& q_item_ip : rivermax_rx_settings["local_ip_addresses"]) {
    rivermax_rx_config.local_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rivermax_rx_settings["source_ip_addresses"]) {
    rivermax_rx_config.source_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rivermax_rx_settings["destination_ip_addresses"]) {
    rivermax_rx_config.destination_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rivermax_rx_settings["destination_ports"]) {
    rivermax_rx_config.destination_ports.emplace_back(q_item_ip.as<uint16_t>());
  }

  rivermax_rx_config.ext_seq_num = rivermax_rx_settings["ext_seq_num"].as<bool>(true);
  rivermax_rx_config.max_path_differential_us = rivermax_rx_settings["max_path_diff_us"].as<uint32_t>(0);
  rivermax_rx_config.allocator_type = rivermax_rx_settings["allocator_type"].as<std::string>("auto");
  rivermax_rx_config.memory_registration = rivermax_rx_settings["memory_registration"].as<bool>(false);
  rivermax_rx_config.sleep_between_operations_us =
      rivermax_rx_settings["sleep_between_operations_us"].as<int>(0);
  rivermax_rx_config.print_parameters = rivermax_rx_settings["verbose"].as<bool>(false);
  rivermax_rx_config.num_of_threads = rivermax_rx_settings["num_of_threads"].as<size_t>(1);
  rivermax_rx_config.send_packet_ext_info = rivermax_rx_settings["send_packet_ext_info"].as<bool>(true);
  rivermax_rx_config.max_chunk_size = q_item["batch_size"].as<size_t>(1024);
  rivermax_rx_config.rx_stats_period_report_ms =
      rivermax_rx_settings["rx_stats_period_report_ms"].as<uint32_t>(0);
  return Status::SUCCESS;
}

Status RivermaxConfigParser::parse_tx_queue_rivermax_config(const YAML::Node& q_item,
                                                        TxQueueConfig& q) {
  return Status::SUCCESS;
}

void RivermaxRxQueueConfig::dump_parameters() const {
  if (this->print_parameters) {
    HOLOSCAN_LOG_INFO("Rivermax RX Queue Config:");
    HOLOSCAN_LOG_INFO("\tNetwork settings:");
    HOLOSCAN_LOG_INFO("\t\tlocal_ips: {}", fmt::join(local_ips, ", "));
    HOLOSCAN_LOG_INFO("\t\tsource_ips: {}", fmt::join(source_ips, ", "));
    HOLOSCAN_LOG_INFO("\t\tdestination_ips: {}", fmt::join(destination_ips, ", "));
    HOLOSCAN_LOG_INFO("\t\tdestination_ports: {}", fmt::join(destination_ports, ", "));
    HOLOSCAN_LOG_INFO("\tGPU settings:");
    HOLOSCAN_LOG_INFO("\t\tGPU ID: {}", gpu_device_id);
    HOLOSCAN_LOG_INFO("\t\tGPU Direct: {}", gpu_direct);
    HOLOSCAN_LOG_INFO("\tMemory config settings:");
    HOLOSCAN_LOG_INFO("\t\tallocator_type: {}", allocator_type);
    HOLOSCAN_LOG_INFO("\t\tmemory_registration: {}", memory_registration);
    HOLOSCAN_LOG_INFO("\tPacket settings:");
    HOLOSCAN_LOG_INFO("\t\tbatch_size/max_chunk_size: {}", max_chunk_size);
    HOLOSCAN_LOG_INFO("\t\tsplit_boundary/header_size: {}", split_boundary);
    HOLOSCAN_LOG_INFO("\t\tmax_packet_size: {}", max_packet_size);
    HOLOSCAN_LOG_INFO("\t\tpackets_buffers_size: {}", packets_buffers_size);
    HOLOSCAN_LOG_INFO("\tRMAX IPO settings:");
    HOLOSCAN_LOG_INFO("\t\text_seq_num: {}", ext_seq_num);
    HOLOSCAN_LOG_INFO("\t\tsleep_between_operations_us: {}", sleep_between_operations_us);
    HOLOSCAN_LOG_INFO("\t\tmax_path_differential_us: {}", max_path_differential_us);
    HOLOSCAN_LOG_INFO("\t\tnum_of_threads: {}", num_of_threads);
    HOLOSCAN_LOG_INFO("\t\tsend_packet_ext_info: {}", send_packet_ext_info);
    HOLOSCAN_LOG_INFO("\t\trx_stats_period_report_ms: {}", rx_stats_period_report_ms);
  }
}

}  // namespace holoscan::advanced_network
