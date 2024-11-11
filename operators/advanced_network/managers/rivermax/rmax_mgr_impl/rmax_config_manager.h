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

#ifndef RMAX_CONFIG_MANAGER_H_
#define RMAX_CONFIG_MANAGER_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "adv_network_mgr.h"
#include "rmax_ano_data_types.h"
#include "rmax_ipo_receiver_service.h"

namespace holoscan::ops {

using namespace ral::services::rmax_ipo_receiver;

/**
 * @brief Configuration structure for Rmax RX queue.
 *
 * This structure holds the configuration settings for an Rmax RX queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RmaxRxQueueConfig : public AnoMgrExtraQueueConfig {
  uint16_t max_packet_size = 0;
  size_t max_chunk_size;
  size_t packets_buffers_size;
  bool gpu_direct;
  int gpu_device_id;
  uint16_t split_boundary;
  std::vector<std::string> local_ips;
  std::vector<std::string> source_ips;
  std::vector<std::string> destination_ips;
  std::vector<uint16_t> destination_ports;
  size_t num_of_threads;
  bool print_parameters;
  uint32_t max_path_differential_us;
  int sleep_between_operations_us;
  std::string allocator_type;
  bool ext_seq_num;
  bool memory_registration;
  bool send_packet_ext_info;
  uint32_t rx_stats_period_report_ms;

 public:
  RmaxRxQueueConfig() = default;
  ~RmaxRxQueueConfig() = default;

  RmaxRxQueueConfig(const RmaxRxQueueConfig& other)
      : AnoMgrExtraQueueConfig(other),
        max_packet_size(other.max_packet_size),
        max_chunk_size(other.max_chunk_size),
        packets_buffers_size(other.packets_buffers_size),
        gpu_direct(other.gpu_direct),
        gpu_device_id(other.gpu_device_id),
        split_boundary(other.split_boundary),
        local_ips(other.local_ips),
        source_ips(other.source_ips),
        destination_ips(other.destination_ips),
        destination_ports(other.destination_ports),
        num_of_threads(other.num_of_threads),
        print_parameters(other.print_parameters),
        max_path_differential_us(other.max_path_differential_us),
        sleep_between_operations_us(other.sleep_between_operations_us),
        allocator_type(other.allocator_type),
        ext_seq_num(other.ext_seq_num),
        memory_registration(other.memory_registration),
        send_packet_ext_info(other.send_packet_ext_info),
        rx_stats_period_report_ms(other.rx_stats_period_report_ms) {}

  RmaxRxQueueConfig& operator=(const RmaxRxQueueConfig& other) {
    if (this != &other) {
      AnoMgrExtraQueueConfig::operator=(other);
      max_packet_size = other.max_packet_size;
      max_chunk_size = other.max_chunk_size;
      packets_buffers_size = other.packets_buffers_size;
      gpu_direct = other.gpu_direct;
      gpu_device_id = other.gpu_device_id;
      split_boundary = other.split_boundary;
      local_ips = other.local_ips;
      source_ips = other.source_ips;
      destination_ips = other.destination_ips;
      destination_ports = other.destination_ports;
      num_of_threads = other.num_of_threads;
      print_parameters = other.print_parameters;
      max_path_differential_us = other.max_path_differential_us;
      sleep_between_operations_us = other.sleep_between_operations_us;
      allocator_type = other.allocator_type;
      ext_seq_num = other.ext_seq_num;
      memory_registration = other.memory_registration;
      send_packet_ext_info = other.send_packet_ext_info;
      rx_stats_period_report_ms = other.rx_stats_period_report_ms;
    }
    return *this;
  }

  void dump_parameters() const;
};

/**
 * @brief Extended configuration for Rmax IPO Receiver.
 */
struct ExtRmaxIPOReceiverConfig : RmaxIPOReceiverConfig {
  bool send_packet_ext_info;
};

/**
 * @brief Base interface for configuration managers.
 *
 * The IConfigManager interface defines the basic operations for configuration managers
 * in Rmax. It provides a template for iterators and a method to set the configuration.
 */
class IConfigManager {
 public:
  static constexpr uint16_t MAX_RMAX_MEMORY_REGIONS = 2;

  template <typename T>
  using ConstIterator = typename std::unordered_map<uint32_t, T>::const_iterator;

  virtual ~IConfigManager() = default;

  /**
   * @brief Sets the configuration.
   *
   * @param cfg The configuration object parsed from YAML.
   * @param rmax_apps_lib Shared pointer to the RmaxAppsLibFacade.
   * @return True if the configuration was successfully set, false otherwise.
   */
  virtual bool set_configuration(const AdvNetConfigYaml& cfg,
                                 std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib) = 0;
};

/**
 * @brief Interface for RX configuration managers.
 *
 * The IRxConfigManager interface extends IConfigManager and defines additional operations
 * specific to RX configuration managers in Rmax.
 */
class IRxConfigManager : public IConfigManager {
 public:
  using ConstIterator = IConfigManager::ConstIterator<ExtRmaxIPOReceiverConfig>;

  /**
   * @brief Gets the beginning iterator for RX configurations.
   *
   * @return The beginning iterator.
   */
  virtual ConstIterator begin() const = 0;

  /**
   * @brief Gets the ending iterator for RX configurations.
   *
   * @return The ending iterator.
   */
  virtual ConstIterator end() const = 0;

  /**
   * @brief Appends a candidate for RX queue configuration.
   *
   * @param port_id The port ID.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  virtual bool append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) = 0;
};

/**
 * @brief Interface for TX configuration managers.
 *
 * The ITxConfigManager interface extends IConfigManager and defines additional operations
 * specific to TX configuration managers in Rmax.
 */
class ITxConfigManager : public IConfigManager {
 public:
  using ConstIterator = IConfigManager::ConstIterator<RmaxBaseServiceConfig>;

  /**
   * @brief Gets the beginning iterator for TX configurations.
   *
   * @return The beginning iterator.
   */
  virtual ConstIterator begin() const = 0;

  /**
   * @brief Gets the ending iterator for TX configurations.
   *
   * @return The ending iterator.
   */
  virtual ConstIterator end() const = 0;

  /**
   * @brief Appends a candidate for TX queue configuration.
   *
   * @param port_id The port ID.
   * @param q The TX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  virtual bool append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) = 0;
};

/**
 * @brief Manages the RX configuration for Rmax.
 *
 * The RxConfigManager class is responsible for managing the configuration settings
 * for RX queues in Rmax. It validates and appends RX queue configurations for given ports.
 */
class RxConfigManager : public IRxConfigManager {
 public:
  using ConstIterator = IConfigManager::ConstIterator<ExtRmaxIPOReceiverConfig>;

  ConstIterator begin() const override { return rx_service_configs_.begin(); }
  ConstIterator end() const override { return rx_service_configs_.end(); }

  /**
   * @brief Sets the configuration for RX queues.
   *
   * @param cfg The configuration object parsed from YAML.
   * @param rmax_apps_lib Shared pointer to the RmaxAppsLibFacade.
   * @return True if the configuration was successfully set, false otherwise.
   */
  bool set_configuration(const AdvNetConfigYaml& cfg,
                         std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib) override {
    cfg_ = cfg;
    rmax_apps_lib_ = rmax_apps_lib;
    is_configuration_set_ = true;
    return true;
  }

  /**
   * @brief Validates RX queue for Rmax Queue configuration. If valid, appends the RX queue for a
   * given port.
   *
   * @param port_id The port ID.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q) override;

 private:
  /**
   * @brief Sets the default configuration for an RX service.
   *
   * @param rx_service_cfg The RX service configuration to set defaults for.
   */
  void set_default_config(ExtRmaxIPOReceiverConfig& rx_service_cfg) const;

  /**
   * @brief Builds the Rmax IPO receiver configuration.
   *
   * @param rx_service_cfg The RX service configuration to build.
   * @param rmax_rx_config The Rmax RX queue configuration.
   * @param q The RX queue configuration.
   * @return True if the configuration was successfully built, false otherwise.
   */
  bool build_rmax_ipo_receiver_config(ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                      const RmaxRxQueueConfig& rmax_rx_config,
                                      const RxQueueConfig& q);

  /**
   * @brief Validates the RX queue configuration.
   *
   * @param rmax_rx_config The Rmax RX queue configuration to validate.
   * @return True if the configuration is valid, false otherwise.
   */
  bool validate_rx_queue_config(const RmaxRxQueueConfig& rmax_rx_config);

  /**
   * @brief Configures the memory allocator for the RMAX RX queue.
   *
   * @param rmax_rx_config The RMAX RX queue configuration.
   * @param q The RX queue configuration.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator(RmaxRxQueueConfig& rmax_rx_config, const RxQueueConfig& q);

  /**
   * @brief Configures the memory allocator for a single memory region.
   *        The allocator will be used for both the header and payload memory.
   *
   * @param rmax_rx_config The RMAX RX queue configuration.
   * @param q The RX queue configuration.
   * @param mr The memory region.
   * @return true if the configuration is successful, false otherwise.
   */
  bool config_memory_allocator_from_single_mrs(RmaxRxQueueConfig& rmax_rx_config,
                                               const RxQueueConfig& q, const MemoryRegion& mr);

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

  bool config_memory_allocator_from_dual_mrs(RmaxRxQueueConfig& rmax_rx_config,
                                             const RxQueueConfig& q, const MemoryRegion& mr_header,
                                             const MemoryRegion& mr_payload);
  /**
   * @brief Sets the GPU memory configuration if applicable.
   *
   * @param rmax_rx_config The RMAX RX queue configuration.
   * @param mr The memory region.
   * @return true if the GPU memory configuration is set, false otherwise.
   */
  bool set_gpu_is_in_use_if_applicable(RmaxRxQueueConfig& rmax_rx_config, const MemoryRegion& mr);

  /**
   * @brief Sets the CPU memory configuration.
   *
   * @param rmax_rx_config The RMAX RX queue configuration.
   */
  void set_gpu_is_not_in_use(RmaxRxQueueConfig& rmax_rx_config);

  /**
   * @brief Sets the allocator type based on the memory region.
   *
   * @param rmax_rx_config The RMAX RX queue configuration.
   * @param mr The memory region.
   */
  void set_cpu_allocator_type(RmaxRxQueueConfig& rmax_rx_config, const MemoryRegion& mr);

  /**
   * @brief Validates the RX queue memory regions configuration.
   *
   * @param q The RX queue configuration.
   * @param rmax_rx_config The Rmax RX queue configuration.
   * @return True if the configuration is valid, false otherwise.
   */
  bool validate_memory_regions_config(const RxQueueConfig& q,
                                      const RmaxRxQueueConfig& rmax_rx_config);

  /**
   * @brief Validates the RX queue memory regions configuration for a single memory region.
   *
   * @param q The RX queue configuration.
   * @param rmax_rx_config The Rmax RX queue configuration.
   * @param mr The memory region.
   * @return True if the configuration is valid, false otherwise.
   */
  bool validate_memory_regions_config_from_single_mrs(const RxQueueConfig& q,
                                                      const RmaxRxQueueConfig& rmax_rx_config,
                                                      const MemoryRegion& mr);

  /**
   * @brief Validates the RX queue memory regions configuration for dual memory regions.
   *
   * @param q The RX queue configuration.
   * @param rmax_rx_config The Rmax RX queue configuration.
   * @param mr_header The header memory region.
   * @param mr_payload The payload memory region.
   * @return True if the configuration is valid, false otherwise.
   */
  bool validate_memory_regions_config_from_dual_mrs(const RxQueueConfig& q,
                                                    const RmaxRxQueueConfig& rmax_rx_config,
                                                    const MemoryRegion& mr_header,
                                                    const MemoryRegion& mr_payload);

  /**
   * @brief Sets common application settings for an RX service.
   *
   * @param app_settings_config The application settings configuration to set.
   * @param rmax_rx_config The Rmax RX queue configuration.
   */
  void set_rx_service_common_app_settings(AppSettings& app_settings_config,
                                          const RmaxRxQueueConfig& rmax_rx_config);

  /**
   * @brief Sets the allocator type for the application settings.
   *
   * @param app_settings_config The application settings configuration to set.
   * @param allocator_type The allocator type to set.
   */
  void set_allocator_type(AppSettings& app_settings_config, const std::string& allocator_type);

  /**
   * @brief Parses and sets the cores for the application settings.
   *
   * @param app_settings_config The application settings configuration to set.
   * @param cores The cores to parse and set.
   * @return True if the cores were successfully parsed and set, false otherwise.
   */
  bool parse_and_set_cores(AppSettings& app_settings_config, const std::string& cores);

  /**
   * @brief Sets the IPO receiver settings for an RX service.
   *
   * @param rx_service_cfg The RX service configuration to set.
   * @param rmax_rx_config The Rmax RX queue configuration.
   */
  void set_rx_service_ipo_receiver_settings(ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                            const RmaxRxQueueConfig& rmax_rx_config);

  /**
   * @brief Adds a new RX service configuration.
   *
   * @param rx_service_cfg The RX service configuration to add.
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   */
  void add_new_rx_service_config(const ExtRmaxIPOReceiverConfig& rx_service_cfg, uint16_t port_id,
                                 uint16_t queue_id);

 private:
  std::unordered_map<uint32_t, ExtRmaxIPOReceiverConfig> rx_service_configs_;
  AdvNetConfigYaml cfg_;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib_ = nullptr;
  bool is_configuration_set_ = false;
};

/**
 * @brief Manages the TX configuration for Rmax.
 *
 * The TxConfigManager class is responsible for managing the configuration settings
 * for TX queues in Rmax. It validates and appends TX queue configurations for given ports.
 */
class TxConfigManager : public ITxConfigManager {
 public:
  using ConstIterator = IConfigManager::ConstIterator<RmaxBaseServiceConfig>;

  ConstIterator begin() const override { return tx_service_configs_.begin(); }
  ConstIterator end() const override { return tx_service_configs_.end(); }

  /**
   * @brief Sets the configuration for TX queues.
   *
   * @param cfg The configuration object parsed from YAML.
   * @param rmax_apps_lib Shared pointer to the RmaxAppsLibFacade.
   * @return True if the configuration was successfully set, false otherwise.
   */
  bool set_configuration(const AdvNetConfigYaml& cfg,
                         std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib) override {
    cfg_ = cfg;
    rmax_apps_lib_ = rmax_apps_lib;
    is_configuration_set_ = true;
    return true;
  }

  /**
   * @brief Validates TX queue for Rmax Queue configuration. If valid, appends the TX queue for a
   * given port.
   *
   * @param port_id The port ID.
   * @param q The TX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_candidate_for_tx_queue(uint16_t port_id, const TxQueueConfig& q) override;

 private:
  std::unordered_map<uint32_t, RmaxBaseServiceConfig> tx_service_configs_;
  AdvNetConfigYaml cfg_;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib_ = nullptr;
  bool is_configuration_set_ = false;
};

/**
 * @brief Manages the configuration for Rmax.
 *
 * The RmaxConfigContainer class is responsible for parsing and managing the configuration
 * settings for Rmax via dedicated configuration managers.
 */
class RmaxConfigContainer {
 public:
  enum class ConfigType { RX, TX };

  /**
   * @brief Constructs a new RmaxConfigContainer object.
   * @param rmax_apps_lib Optional shared pointer to the RmaxAppsLibFacade.
   */
  explicit RmaxConfigContainer(std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr)
      : rmax_apps_lib_(rmax_apps_lib) {
    initialize_managers();
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
  bool parse_configuration(const AdvNetConfigYaml& cfg);

  std::shared_ptr<IConfigManager> get_config_manager(ConfigType type) const {
    auto it = config_managers_.find(type);
    if (it != config_managers_.end()) { return it->second; }
    return nullptr;
  }

  /**
   * @brief Gets the current log level for Rmax.
   *
   * @return The current log level.
   */
  RmaxLogLevel::Level get_rmax_log_level() const { return rmax_log_level_; }

 private:
  /**
   * @brief Initializes the configuration managers.
   *
   * This function initializes the configuration managers for RX and TX services.
   */
  void initialize_managers();

  /**
   * @brief Adds a configuration manager.
   *
   * This function adds a configuration manager for the specified type.
   *
   * @param type The type of configuration manager to add.
   * @param config_manager The shared pointer to the configuration manager.
   */
  void add_config_manager(ConfigType type, std::shared_ptr<IConfigManager> config_manager);

  /**
   * @brief Parses the RX queues configuration.
   *
   * This function parses the configuration for RX queues for the specified port ID.
   *
   * @param port_id The port ID for which to parse the RX queues.
   * @param queues The vector of RX queue configurations.
   * @return An integer indicating the success or failure of the parsing operation.
   */
  int parse_rx_queues(uint16_t port_id, const std::vector<RxQueueConfig>& queues);

  /**
   * @brief Parses the TX queues configuration.
   *
   * This function parses the configuration for TX queues for the specified port ID.
   *
   * @param port_id The port ID for which to parse the TX queues.
   * @param queues The vector of TX queue configurations.
   * @return An integer indicating the success or failure of the parsing operation.
   */
  int parse_tx_queues(uint16_t port_id, const std::vector<TxQueueConfig>& queues);

  /**
   * @brief Sets the Rmax log level based on the provided Ano log level.
   *
   * This function converts the provided Ano log level to the corresponding Rmax log level
   * and sets it as the current log level for Rmax.
   *
   * @param level The Ano log level to be converted and set.
   */
  void set_rmax_log_level(AnoLogLevel::Level level) {
    rmax_log_level_ = RmaxLogLevel::from_ano_log_level(level);
  }

 private:
  RmaxLogLevel::Level rmax_log_level_ = RmaxLogLevel::OFF;
  std::unordered_map<ConfigType, std::shared_ptr<IConfigManager>> config_managers_;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib_ = nullptr;
  AdvNetConfigYaml cfg_;
  bool is_configured_ = false;
};

/**
 * @brief Parses the configuration for Rmax.
 *
 * The RmaxConfigParser class is responsible for parsing the configuration settings for Rmax.
 */
class RmaxConfigParser {
 public:
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
  static AdvNetStatus parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q);

  /**
   * @brief Parses the TX queue Rivermax configuration.
   *
   * @param q_item The YAML node containing the queue item.
   * @param q The TX queue configuration to be populated.
   * @return AdvNetStatus indicating the success or failure of the operation.
   */
  static AdvNetStatus parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q);
};

}  // namespace holoscan::ops

#endif  // RMAX_CONFIG_MANAGER_H_
