/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */
#ifndef RMAX_CONFIG_MANAGER_H_
#define RMAX_CONFIG_MANAGER_H_

#include <unordered_map>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

#include "adv_network_mgr.h"
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
  bool sleep_between_operations;
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
        sleep_between_operations(other.sleep_between_operations),
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
      sleep_between_operations = other.sleep_between_operations;
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
 * @brief Manages the configuration for Rmax.
 */
/**
 * @brief Manages the configuration for Rmax.
 *
 * The RmaxConfigManager class is responsible for parsing and managing the configuration
 * settings for Rmax. It handles the configuration of RX queues, validation of configurations,
 * and setting various application settings.
 */
class RmaxConfigManager {
 public:
  static constexpr uint16_t RMAX_DEFAULT_LOG_LEVEL = 3;
  static constexpr uint16_t RMAX_MIN_LOG_LEVEL = 1;
  static constexpr uint16_t RMAX_MAX_LOG_LEVEL = 6;
  static constexpr uint16_t MAX_RMAX_MEMORY_REGIONS = 2;

  /**
   * @brief Constructs a new RmaxConfigManager object.
   * @param rmax_apps_lib Optional shared pointer to the RmaxAppsLibFacade.
   */
  RmaxConfigManager(std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr)
      : rmax_apps_lib_(rmax_apps_lib) {}

  /**
   * @brief Parses the configuration from a YAML file.
   *
   * @param cfg The configuration object parsed from YAML.
   * @return True if the configuration was successfully parsed, false otherwise.
   */
  bool parse_configuration(const AdvNetConfigYaml& cfg);

  /**
   * @brief Gets the current log level for Rmax.
   *
   * @return The current log level.
   */
  uint16_t get_rmax_log_level() const { return rmax_log_level_; }

  /**
   * @brief Gets the RX service configurations.
   *
   * @return A constant reference to the unordered map of RX service configurations.
   */
  const std::unordered_map<uint32_t, ExtRmaxIPOReceiverConfig>& get_rx_service_configs() const {
    return rx_service_configs_;
  }

 private:
  /**
   * @brief Sets the default configuration for an RX service.
   *
   * @param rx_service_cfg The RX service configuration to set defaults for.
   */
  void set_default_config(ExtRmaxIPOReceiverConfig& rx_service_cfg) const;

  /**
   * @brief Validates RX queue for Rmax Queue configuration. If valid, appends the RX queue for a
   * given port.
   *
   * @param port_id The port ID.
   * @param q The RX queue configuration.
   * @return True if the configuration was appended successfully, false otherwise.
   */
  bool append_candidate_for_rx_queue(uint16_t port_id, const RxQueueConfig& q);

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

  uint16_t rmax_log_level_ = RMAX_DEFAULT_LOG_LEVEL;
  std::unordered_map<uint32_t, ExtRmaxIPOReceiverConfig> rx_service_configs_;
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