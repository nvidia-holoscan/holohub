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
#include "rmax_mgr_impl/burst_manager.h"
#include <holoscan/logger/logger.hpp>

#include "rmax_config_manager.h"

namespace holoscan::ops {

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
  static std::shared_ptr<IConfigManager> create_manager(RmaxConfigContainer::ConfigType type) {
    switch (type) {
      case RmaxConfigContainer::ConfigType::RX:
        return std::make_shared<RxConfigManager>();
      case RmaxConfigContainer::ConfigType::TX:
        return std::make_shared<TxConfigManager>();
      default:
        return nullptr;
    }
  }
};

/**
 * @brief Adds a configuration manager.
 *
 * This function adds a configuration manager for the specified type.
 *
 * @param type The type of configuration manager to add.
 * @param config_manager The shared pointer to the configuration manager.
 */
void RmaxConfigContainer::add_config_manager(ConfigType type,
                                             std::shared_ptr<IConfigManager> config_manager) {
  config_managers_[type] = config_manager;
}

/**
 * @brief Initializes the configuration managers.
 *
 * This function initializes the configuration managers for RX and TX services.
 */
void RmaxConfigContainer::initialize_managers() {
  add_config_manager(ConfigType::RX, ConfigManagerFactory::create_manager(ConfigType::RX));
  add_config_manager(ConfigType::TX, ConfigManagerFactory::create_manager(ConfigType::TX));
}

/**
 * @brief Parses the RX queues configuration.
 *
 * This function parses the configuration for RX queues for the specified port ID.
 *
 * @param port_id The port ID for which to parse the RX queues.
 * @param queues The vector of RX queue configurations.
 * @return An integer indicating the success or failure of the parsing operation.
 */
int RmaxConfigContainer::parse_rx_queues(uint16_t port_id,
                                         const std::vector<RxQueueConfig>& queues) {
  int rmax_rx_config_found = 0;

  auto rx_manager = std::dynamic_pointer_cast<RxConfigManager>(
      get_config_manager(RmaxConfigContainer::ConfigType::RX));

  if (!rx_manager) { return 0; }

  rx_manager->set_configuration(cfg_, rmax_apps_lib_);

  for (const auto& q : queues) {
    if (!rx_manager->append_candidate_for_rx_queue(port_id, q)) { continue; }
    rmax_rx_config_found++;
  }

  return rmax_rx_config_found;
}

/**
 * @brief Parses the TX queues configuration.
 *
 * This function parses the configuration for TX queues for the specified port ID.
 *
 * @param port_id The port ID for which to parse the TX queues.
 * @param queues The vector of TX queue configurations.
 * @return An integer indicating the success or failure of the parsing operation.
 */
int RmaxConfigContainer::parse_tx_queues(uint16_t port_id,
                                         const std::vector<TxQueueConfig>& queues) {
  int rmax_tx_config_found = 0;

  auto tx_manager = std::dynamic_pointer_cast<TxConfigManager>(
      get_config_manager(RmaxConfigContainer::ConfigType::TX));

  if (!tx_manager) { return 0; }

  tx_manager->set_configuration(cfg_, rmax_apps_lib_);

  for (const auto& q : queues) {
    if (!tx_manager->append_candidate_for_tx_queue(port_id, q)) { continue; }
    rmax_tx_config_found++;
  }

  return rmax_tx_config_found;
}

/**
 * @brief Parses the configuration from the YAML file.
 *
 * This function iterates over the interfaces and their respective RX an TX queues
 * defined in the configuration YAML, extracting and validating the necessary
 * settings for each queue. It then populates the RX and TX service configuration
 * structures with these settings. The parsing is done via dedicated configuration managers.
 *
 * @param cfg The configuration YAML.
 * @return True if the configuration was successfully parsed, false otherwise.
 */
bool RmaxConfigContainer::parse_configuration(const AdvNetConfigYaml& cfg) {
  int rmax_rx_config_found = 0;
  int rmax_tx_config_found = 0;

  is_configured_ = false;
  cfg_ = cfg;

  for (const auto& intf : cfg.ifs_) {
    HOLOSCAN_LOG_INFO("Rmax init Port {} -- RX: {} TX: {}",
                      intf.port_id_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");

    rmax_rx_config_found += parse_rx_queues(intf.port_id_, intf.rx_.queues_);
    rmax_tx_config_found += parse_tx_queues(intf.port_id_, intf.tx_.queues_);
  }

  set_rmax_log_level(cfg.log_level_);

  if (rmax_rx_config_found == 0 && rmax_tx_config_found == 0) {
    HOLOSCAN_LOG_ERROR("Failed to parse Rivermax ANO settings. No valid settings found");
    return false;
  }

  HOLOSCAN_LOG_INFO(
      "Rivermax ANO settings were successfully parsed, Found {} RX Queues and {} TX Queues "
      "settings",
      rmax_rx_config_found,
      rmax_tx_config_found);

  is_configured_ = true;
  return true;
}

/**
 * @brief Sets the default configuration for an RX service.
 *
 * @param rx_service_cfg The RX service configuration to be set.
 */
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

/**
 * @brief Validates RX queue for Rmax Queue configuration. If valid, appends the RX queue for a
 * given port.
 *
 * @param port_id The port ID.
 * @param q The RX queue configuration.
 * @return True if the configuration was appended successfully, false otherwise.
 */
bool RxConfigManager::append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) {
  HOLOSCAN_LOG_INFO(
      "Configuring RX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, port_id);

  if (is_configuration_set_ == false) {
    HOLOSCAN_LOG_ERROR("Configuration wasn't set for RxConfigManger");
    return false;
  }

  // extra queue config_ contains RMAX configuration. If it is not set, return false
  if (!q.common_.extra_queue_config_) return false;

  auto* rmax_rx_config_ptr = dynamic_cast<RmaxRxQueueConfig*>(q.common_.extra_queue_config_);
  if (!rmax_rx_config_ptr) {
    HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RmaxRxQueueConfig");
    return false;
  }

  RmaxRxQueueConfig rmax_rx_config(*rmax_rx_config_ptr);

  if (!validate_rx_queue_config(rmax_rx_config)) { return false; }

  if (!validate_memory_regions_config(q, rmax_rx_config)) { return false; }

  if (config_memory_allocator(rmax_rx_config, q) == false) { return false; }

  rmax_rx_config.dump_parameters();

  ExtRmaxIPOReceiverConfig rx_service_cfg;

  if (!build_rmax_ipo_receiver_config(rx_service_cfg, rmax_rx_config, q)) { return false; }

  add_new_rx_service_config(rx_service_cfg, port_id, q.common_.id_);

  return true;
}

/**
 * @brief Configures the memory allocator for the RMAX RX queue.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 * @param q The RX queue configuration.
 * @return true if the configuration is successful, false otherwise.
 */
bool RxConfigManager::config_memory_allocator(RmaxRxQueueConfig& rmax_rx_config,
                                              const RxQueueConfig& q) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  HOLOSCAN_LOG_INFO(
      "Configuring memory allocator for RMAX RX queue: {}, number of memory regions: {}",
      q.common_.name_,
      num_of_mrs);
  if (num_of_mrs == 1) {
    return config_memory_allocator_from_single_mrs(rmax_rx_config, q, cfg_.mrs_[q.common_.mrs_[0]]);
  } else if (num_of_mrs == 2) {
    return config_memory_allocator_from_dual_mrs(
        rmax_rx_config, q, cfg_.mrs_[q.common_.mrs_[0]], cfg_.mrs_[q.common_.mrs_[1]]);
  } else {
    HOLOSCAN_LOG_ERROR("Incompatible number of memory regions for Rivermax RX queue: {} [1..{}]",
                       num_of_mrs,
                       MAX_RMAX_MEMORY_REGIONS);
    return false;
  }
}

/**
 * @brief Configures the memory allocator for a single memory region.
 *        The allocator will be used for both the header and payload memory.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 * @param q The RX queue configuration.
 * @param mr The memory region.
 * @return true if the configuration is successful, false otherwise.
 */
bool RxConfigManager::config_memory_allocator_from_single_mrs(RmaxRxQueueConfig& rmax_rx_config,
                                                              const RxQueueConfig& q,
                                                              const MemoryRegion& mr) {
  rmax_rx_config.split_boundary = 0;
  rmax_rx_config.max_packet_size = mr.buf_size_;
  rmax_rx_config.packets_buffers_size = mr.num_bufs_;

  if (set_gpu_is_in_use_if_applicable(rmax_rx_config, mr)) { return true; }

  set_gpu_is_not_in_use(rmax_rx_config);
  set_cpu_allocator_type(rmax_rx_config, mr);

  return true;
}

/**
 * @brief Configures the memory allocator for dual memory regions.
 *        If GPU is in use, it will be used for the payload memory region,
 *        and the CPU allocator will be used for the header memory region.
 *        Otherwise, the function expects that the same allocator is configured
 *        for both memory regions.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 * @param q The RX queue configuration.
 * @param mr_header The header memory region.
 * @param mr_payload The payload memory region.
 * @return true if the configuration is successful, false otherwise.
 */
bool RxConfigManager::config_memory_allocator_from_dual_mrs(RmaxRxQueueConfig& rmax_rx_config,
                                                            const RxQueueConfig& q,
                                                            const MemoryRegion& mr_header,
                                                            const MemoryRegion& mr_payload) {
  rmax_rx_config.split_boundary = mr_header.buf_size_;
  rmax_rx_config.max_packet_size = mr_payload.buf_size_;
  rmax_rx_config.packets_buffers_size = mr_payload.num_bufs_;

  if (!set_gpu_is_in_use_if_applicable(rmax_rx_config, mr_payload)) {
    set_gpu_is_not_in_use(rmax_rx_config);
  }

  set_cpu_allocator_type(rmax_rx_config, mr_header);

  return true;
}

/**
 * @brief Sets the GPU memory configuration if applicable.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 * @param mr The memory region.
 * @return true if the GPU memory configuration is set, false otherwise.
 */
bool RxConfigManager::set_gpu_is_in_use_if_applicable(RmaxRxQueueConfig& rmax_rx_config,
                                                      const MemoryRegion& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::DEVICE || mr.kind_ == MemoryKind::HOST_PINNED) {
#else
  if (mr.kind_ == MemoryKind::DEVICE) {
#endif
    rmax_rx_config.gpu_device_id = mr.affinity_;
    rmax_rx_config.gpu_direct = true;
    return true;
  }
  return false;
}

/**
 * @brief Sets the CPU memory configuration.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 */
void RxConfigManager::set_gpu_is_not_in_use(RmaxRxQueueConfig& rmax_rx_config) {
  rmax_rx_config.gpu_device_id = -1;
  rmax_rx_config.gpu_direct = false;
}

/**
 * @brief Sets the allocator type based on the memory region.
 *
 * @param rmax_rx_config The RMAX RX queue configuration.
 * @param mr The memory region.
 */
void RxConfigManager::set_cpu_allocator_type(RmaxRxQueueConfig& rmax_rx_config,
                                             const MemoryRegion& mr) {
#if RMAX_TEGRA
  if (mr.kind_ == MemoryKind::HOST) {
#else
  if (mr.kind_ == MemoryKind::HOST || mr.kind_ == MemoryKind::HOST_PINNED) {
#endif

    rmax_rx_config.allocator_type = "malloc";
  } else if (mr.kind_ == MemoryKind::HUGE) {
    if (rmax_rx_config.allocator_type != "huge_page_default" &&
        rmax_rx_config.allocator_type != "huge_page_2mb" &&
        rmax_rx_config.allocator_type != "huge_page_512mb" &&
        rmax_rx_config.allocator_type != "huge_page_1gb") {
      rmax_rx_config.allocator_type = "huge_page_default";
    }  // else the allocator type is already set
  }
}

/**
 * @brief Validates the RX queue memory regions configuration.
 *
 * @param q The RX queue configuration.
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @return True if the configuration is valid, false otherwise.
 */
bool RxConfigManager::validate_memory_regions_config(const RxQueueConfig& q,
                                                     const RmaxRxQueueConfig& rmax_rx_config) {
  uint16_t num_of_mrs = q.common_.mrs_.size();
  try {
    if (num_of_mrs == 1) {
      return validate_memory_regions_config_from_single_mrs(
          q, rmax_rx_config, cfg_.mrs_.at(q.common_.mrs_[0]));
    } else if (num_of_mrs == 2) {
      return validate_memory_regions_config_from_dual_mrs(
          q, rmax_rx_config, cfg_.mrs_.at(q.common_.mrs_[0]), cfg_.mrs_.at(q.common_.mrs_[1]));
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

/**
 * @brief Validates the RX queue memory regions configuration for a single memory region.
 *
 * @param q The RX queue configuration.
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @param mr The memory region.
 * @return True if the configuration is valid, false otherwise.
 */
bool RxConfigManager::validate_memory_regions_config_from_single_mrs(
    const RxQueueConfig& q, const RmaxRxQueueConfig& rmax_rx_config, const MemoryRegion& mr) {
  return true;
}

/**
 * @brief Validates the RX queue memory regions configuration for dual memory regions.
 *
 * @param q The RX queue configuration.
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @param mr_header The header memory region.
 * @param mr_payload The payload memory region.
 * @return True if the configuration is valid, false otherwise.
 */
bool RxConfigManager::validate_memory_regions_config_from_dual_mrs(
    const RxQueueConfig& q, const RmaxRxQueueConfig& rmax_rx_config, const MemoryRegion& mr_header,
    const MemoryRegion& mr_payload) {
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

/**
 * @brief Builds the Rmax IPO receiver configuration.
 *
 * @param rx_service_cfg The RX service configuration to be built.
s * @param rmax_rx_config The Rmax RX queue configuration.
 * @param q The RX queue configuration.
 * @return True if the configuration is successful, false otherwise.
 */
bool RxConfigManager::build_rmax_ipo_receiver_config(ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                                     const RmaxRxQueueConfig& rmax_rx_config,
                                                     const RxQueueConfig& q) {
  rx_service_cfg.app_settings = std::make_shared<AppSettings>();
  set_default_config(rx_service_cfg);

  auto& app_settings_config = *(rx_service_cfg.app_settings);

  set_rx_service_common_app_settings(app_settings_config, rmax_rx_config);

  if (!parse_and_set_cores(app_settings_config, q.common_.cpu_core_)) { return false; }

  set_rx_service_ipo_receiver_settings(rx_service_cfg, rmax_rx_config);

  return true;
}

/**
 * @brief Validates the RX queue configuration.
 *
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @return True if the configuration is valid, false otherwise.
 */
bool RxConfigManager::validate_rx_queue_config(const RmaxRxQueueConfig& rmax_rx_config) {
  if (rmax_rx_config.source_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Source IP addresses are not set for RTP stream");
    return false;
  }

  if (rmax_rx_config.local_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Local IP addresses are not set for RTP stream");
    return false;
  }

  if (rmax_rx_config.destination_ips.empty()) {
    HOLOSCAN_LOG_ERROR("Destination IP addresses are not set for RTP stream");
    return false;
  }

  if (rmax_rx_config.destination_ports.empty()) {
    HOLOSCAN_LOG_ERROR("Destination ports are not set for RTP stream");
    return false;
  }

  if ((rmax_rx_config.local_ips.size() != rmax_rx_config.source_ips.size()) ||
      (rmax_rx_config.local_ips.size() != rmax_rx_config.destination_ips.size()) ||
      (rmax_rx_config.local_ips.size() != rmax_rx_config.destination_ports.size())) {
    HOLOSCAN_LOG_ERROR(
        "Local/Source/Destination IP addresses and ports sizes are not equal for RTP stream");
    return false;
  }

  return true;
}

/**
 * @brief Sets the common application settings for the RX service.
 *
 * @param app_settings_config The application settings configuration.
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @param split_boundary The split boundary value.
 */
void RxConfigManager::set_rx_service_common_app_settings(AppSettings& app_settings_config,
                                                         const RmaxRxQueueConfig& rmax_rx_config) {
  app_settings_config.local_ips = rmax_rx_config.local_ips;
  app_settings_config.source_ips = rmax_rx_config.source_ips;
  app_settings_config.destination_ips = rmax_rx_config.destination_ips;
  app_settings_config.destination_ports = rmax_rx_config.destination_ports;

  if (rmax_rx_config.gpu_direct) {
    app_settings_config.gpu_id = rmax_rx_config.gpu_device_id;
  } else {
    app_settings_config.gpu_id = INVALID_GPU_ID;
  }

  set_allocator_type(app_settings_config, rmax_rx_config.allocator_type);

  if (cfg_.common_.master_core_ >= 0 &&
      cfg_.common_.master_core_ < std::thread::hardware_concurrency()) {
    app_settings_config.internal_thread_core = cfg_.common_.master_core_;
  } else {
    app_settings_config.internal_thread_core = CPU_NONE;
  }
  app_settings_config.num_of_threads = rmax_rx_config.num_of_threads;
  app_settings_config.print_parameters = rmax_rx_config.print_parameters;
  app_settings_config.sleep_between_operations_us = rmax_rx_config.sleep_between_operations_us;
  app_settings_config.packet_payload_size = rmax_rx_config.max_packet_size;
  app_settings_config.packet_app_header_size = rmax_rx_config.split_boundary;
  app_settings_config.num_of_packets_in_chunk =
      std::pow(2, std::ceil(std::log2(rmax_rx_config.packets_buffers_size)));
}

/**
 * @brief Sets the allocator type for the application settings.
 *
 * @param app_settings_config The application settings configuration.
 * @param allocator_type The allocator type string.
 */
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

/**
 * @brief Parses and sets the cores for the application settings.
 *
 * @param app_settings_config The application settings configuration.
 * @param cores The cores configuration string.
 * @return True if the cores are successfully parsed and set, false otherwise.
 */
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

/**
 * @brief Sets the IPO receiver settings for the RX service.
 *
 * @param rx_service_cfg The RX service configuration.
 * @param rmax_rx_config The Rmax RX queue configuration.
 */
void RxConfigManager::set_rx_service_ipo_receiver_settings(
    ExtRmaxIPOReceiverConfig& rx_service_cfg, const RmaxRxQueueConfig& rmax_rx_config) {
  rx_service_cfg.is_extended_sequence_number = rmax_rx_config.ext_seq_num;
  rx_service_cfg.max_path_differential_us = rmax_rx_config.max_path_differential_us;
  if (rx_service_cfg.max_path_differential_us >= USECS_IN_SECOND) {
    HOLOSCAN_LOG_ERROR("Max path differential must be less than 1 second");
    rx_service_cfg.max_path_differential_us = USECS_IN_SECOND;
  }

  rx_service_cfg.rx_stats_period_report_ms = rmax_rx_config.rx_stats_period_report_ms;
  rx_service_cfg.register_memory = rmax_rx_config.memory_registration;
  rx_service_cfg.max_chunk_size = rmax_rx_config.max_chunk_size;
  rx_service_cfg.rmax_apps_lib = this->rmax_apps_lib_;

  rx_service_cfg.send_packet_ext_info = rmax_rx_config.send_packet_ext_info;
}

/**
 * @brief Adds a new RX service configuration to the configuration map.
 *
 * @param rx_service_cfg The RX service configuration.
 * @param port_id The port ID.
 * @param queue_id The queue ID.
 */
void RxConfigManager::add_new_rx_service_config(const ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                                uint16_t port_id, uint16_t queue_id) {
  uint32_t key = RmaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  if (rx_service_configs_.find(key) != rx_service_configs_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Rivermax ANO settings for port {} and queue {} already exists", port_id, queue_id);
    return;
  }
  HOLOSCAN_LOG_INFO("Rivermax ANO settings for port {} and queue {} added", port_id, queue_id);

  rx_service_configs_[key] = rx_service_cfg;
}

/**
 * @brief Validates TX queue for Rmax Queue configuration. If valid, appends the TX queue for a
 * given port.
 *
 * @param port_id The port ID.
 * @param q The TX queue configuration.
 * @return True if the configuration was appended successfully, false otherwise.
 */
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

/**
 * @brief Parses the RX queue configuration from a YAML node.
 *
 * This function extracts the RX queue configuration settings from the provided YAML node
 * and populates the RxQueueConfig structure with the extracted values.
 *
 * @param q_item The YAML node containing the RX queue configuration.
 * @param q The RxQueueConfig structure to be populated.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxConfigParser::parse_rx_queue_rivermax_config(const YAML::Node& q_item,
                                                              RxQueueConfig& q) {
  const auto& rmax_rx_settings = q_item["rmax_rx_settings"];

  if (!rmax_rx_settings) {
    HOLOSCAN_LOG_ERROR("Rmax RX settings not found");
    return AdvNetStatus::INVALID_PARAMETER;
  }

  q.common_.extra_queue_config_ = new RmaxRxQueueConfig();
  auto& rmax_rx_config = *(reinterpret_cast<RmaxRxQueueConfig*>(q.common_.extra_queue_config_));

  for (const auto& q_item_ip : rmax_rx_settings["local_ip_addresses"]) {
    rmax_rx_config.local_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rmax_rx_settings["source_ip_addresses"]) {
    rmax_rx_config.source_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rmax_rx_settings["destination_ip_addresses"]) {
    rmax_rx_config.destination_ips.emplace_back(q_item_ip.as<std::string>());
  }

  for (const auto& q_item_ip : rmax_rx_settings["destination_ports"]) {
    rmax_rx_config.destination_ports.emplace_back(q_item_ip.as<uint16_t>());
  }

  rmax_rx_config.ext_seq_num = rmax_rx_settings["ext_seq_num"].as<bool>(true);
  rmax_rx_config.max_path_differential_us = rmax_rx_settings["max_path_diff_us"].as<uint32_t>(0);
  rmax_rx_config.allocator_type = rmax_rx_settings["allocator_type"].as<std::string>("auto");
  rmax_rx_config.memory_registration = rmax_rx_settings["memory_registration"].as<bool>(false);
  rmax_rx_config.sleep_between_operations_us =
      rmax_rx_settings["sleep_between_operations_us"].as<int>(0);
  rmax_rx_config.print_parameters = rmax_rx_settings["verbose"].as<bool>(false);
  rmax_rx_config.num_of_threads = rmax_rx_settings["num_of_threads"].as<size_t>(1);
  rmax_rx_config.send_packet_ext_info = rmax_rx_settings["send_packet_ext_info"].as<bool>(true);
  rmax_rx_config.max_chunk_size = q_item["batch_size"].as<size_t>(1024);
  rmax_rx_config.rx_stats_period_report_ms =
      rmax_rx_settings["rx_stats_period_report_ms"].as<uint32_t>(0);
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Parses the TX queue Rivermax configuration.
 *
 * @param q_item The YAML node containing the queue item.
 * @param q The TX queue configuration to be populated.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxConfigParser::parse_tx_queue_rivermax_config(const YAML::Node& q_item,
                                                              TxQueueConfig& q) {
  return AdvNetStatus::SUCCESS;
}

void RmaxRxQueueConfig::dump_parameters() const {
  if (this->print_parameters) {
    HOLOSCAN_LOG_INFO("Rivermax RX Queue Config:");
    // print gpu settings
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

}  // namespace holoscan::ops
