#include "rt_threads.h"
#include "rmax_ipo_receiver_service.h"
#include "rmax_mgr_impl/burst_manager.h"
#include <holoscan/logger/logger.hpp>

#include "rmax_config_manager.h"

namespace holoscan::ops {

static constexpr int USECS_IN_SECOND = 1000000;

/**
 * @brief Sets the default configuration for an RX service.
 *
 * @param rx_service_cfg The RX service configuration to be set.
 */
void RmaxConfigManager::set_default_config(ExtRmaxIPOReceiverConfig& rx_service_cfg) {
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
 * @brief Parses the configuration from the YAML file.
 *
 * This function iterates over the interfaces and their respective RX queues
 * defined in the configuration YAML, extracting and validating the necessary
 * settings for each RX queue. It then populates the RX service configuration
 * structures with these settings.
 *
 * @param cfg The configuration YAML.
 * @param rmax_apps_lib Optional shared pointer to the RmaxAppsLibFacade.
 * @return True if the configuration was successfully parsed, false otherwise.
 */
bool RmaxConfigManager::parse_configuration(
    const AdvNetConfigYaml& cfg, std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib) {
  int rmax_rx_config_found = 0;

  this->rmax_apps_lib = rmax_apps_lib;

  for (const auto& intf : cfg.ifs_) {
    HOLOSCAN_LOG_INFO("Rmax init Port {} -- RX: {} TX: {}",
                      intf.port_id_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");

    for (const auto& q : intf.rx_.queues_) {
      if (!configure_rx_queue(intf.port_id_, cfg.common_.master_core_, q)) { continue; }
      rmax_rx_config_found++;
    }
  }

  if (cfg.debug_ >= RMAX_MIN_LOG_LEVEL && cfg.debug_ <= RMAX_MAX_LOG_LEVEL) {
    rmax_log_level = cfg.debug_;
  }

  if (rmax_rx_config_found > 0) {
    HOLOSCAN_LOG_INFO(
        "RMAX ANO settings were successfully parsed, Found {} RMAX RX Queues settings",
        rmax_rx_config_found);
  } else {
    HOLOSCAN_LOG_ERROR("Failed to parse RMAX ANO settings. No valid settings found");
    return false;
  }

  return true;
}

/**
 * @brief Configures the RX queue for a given port.
 *
 * @param port_id The port ID.
 * @param master_core The master core ID.
 * @param q The RX queue configuration.
 * @return True if the configuration is successful, false otherwise.
 */
bool RmaxConfigManager::configure_rx_queue(uint16_t port_id, int master_core,
                                           const RxQueueConfig& q) {
  HOLOSCAN_LOG_INFO(
      "Configuring RX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, port_id);

  // extra queue config_ contains RMAX configuration. If it is not set, return false
  if (!q.common_.extra_queue_config_) return false;

  auto* rmax_rx_config_ptr = dynamic_cast<RmaxRxQueueConfig*>(q.common_.extra_queue_config_);
  if (!rmax_rx_config_ptr) {
    HOLOSCAN_LOG_ERROR("Failed to cast extra queue config to RmaxRxQueueConfig");
    return false;
  }

  const auto& rmax_rx_config = *rmax_rx_config_ptr;
  if (!validate_rx_queue_config(rmax_rx_config)) { return false; }

  ExtRmaxIPOReceiverConfig rx_service_cfg;

  if (!build_rmax_ipo_receiver_config(rmax_rx_config,
                                      master_core,
                                      q.common_.cpu_core_,
                                      q.common_.split_boundary_,
                                      rx_service_cfg)) {
    return false;
  }

  add_new_rx_service_config(rx_service_cfg, port_id, q.common_.id_);

  return true;
}

/**
 * @brief Builds the Rmax IPO receiver configuration.
 *
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @param master_core The master core ID.
 * @param cores The cores configuration string.
 * @param split_boundary The split boundary value.
 * @param rx_service_cfg The RX service configuration to be built.
 * @return True if the configuration is successful, false otherwise.
 */
bool RmaxConfigManager::build_rmax_ipo_receiver_config(const RmaxRxQueueConfig& rmax_rx_config,
                                                       int master_core, const std::string& cores,
                                                       int split_boundary,
                                                       ExtRmaxIPOReceiverConfig& rx_service_cfg) {
  rx_service_cfg.app_settings = std::make_shared<AppSettings>();
  set_default_config(rx_service_cfg);

  auto& app_settings_config = *(rx_service_cfg.app_settings);
  set_rx_service_common_app_settings(
      app_settings_config, rmax_rx_config, master_core, split_boundary);

  if (!parse_and_set_cores(app_settings_config, cores)) { return false; }

  set_rx_service_ipo_receiver_settings(rx_service_cfg, rmax_rx_config);

  return true;
}

/**
 * @brief Validates the RX queue configuration.
 *
 * @param rmax_rx_config The Rmax RX queue configuration.
 * @return True if the configuration is valid, false otherwise.
 */
bool RmaxConfigManager::validate_rx_queue_config(const RmaxRxQueueConfig& rmax_rx_config) {
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
 * @param master_core The master core ID.
 * @param split_boundary The split boundary value.
 */
void RmaxConfigManager::set_rx_service_common_app_settings(AppSettings& app_settings_config,
                                                           const RmaxRxQueueConfig& rmax_rx_config,
                                                           int master_core, int split_boundary) {
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

  if (master_core >= 0 && master_core < std::thread::hardware_concurrency()) {
    app_settings_config.internal_thread_core = master_core;
  } else {
    app_settings_config.internal_thread_core = CPU_NONE;
  }
  app_settings_config.num_of_threads = rmax_rx_config.num_of_threads;
  app_settings_config.print_parameters = rmax_rx_config.print_parameters;
  app_settings_config.sleep_between_operations_us = rmax_rx_config.sleep_between_operations_us;
  app_settings_config.packet_payload_size = rmax_rx_config.max_packet_size;
  app_settings_config.packet_app_header_size = split_boundary;
  app_settings_config.num_of_chunks = rmax_rx_config.num_concurrent_batches;
  app_settings_config.num_of_packets_in_chunk =
      rmax_rx_config.max_chunk_size * rmax_rx_config.num_concurrent_batches;
}

/**
 * @brief Sets the allocator type for the application settings.
 *
 * @param app_settings_config The application settings configuration.
 * @param allocator_type The allocator type string.
 */
void RmaxConfigManager::set_allocator_type(AppSettings& app_settings_config,
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
bool RmaxConfigManager::parse_and_set_cores(AppSettings& app_settings_config,
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
void RmaxConfigManager::set_rx_service_ipo_receiver_settings(
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
  rx_service_cfg.rmax_apps_lib = this->rmax_apps_lib;

  rx_service_cfg.send_packet_ext_info = rmax_rx_config.send_packet_ext_info;
}

/**
 * @brief Adds a new RX service configuration to the configuration map.
 *
 * @param rx_service_cfg The RX service configuration.
 * @param port_id The port ID.
 * @param queue_id The queue ID.
 */
void RmaxConfigManager::add_new_rx_service_config(const ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                                  uint16_t port_id, uint16_t queue_id) {
  uint32_t key = RmaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  if (rx_service_configs.find(key) != rx_service_configs.end()) {
    HOLOSCAN_LOG_ERROR(
        "RMAX ANO settings for port {} and queue {} already exists", port_id, queue_id);
    return;
  }

  rx_service_configs[key] = rx_service_cfg;
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
AdvNetStatus RmaxConfigManager::parse_rx_queue_rivermax_config(const YAML::Node& q_item,
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
  rmax_rx_config.gpu_direct = rmax_rx_settings["gpu_direct"].as<bool>(false);
  rmax_rx_config.gpu_device_id = rmax_rx_settings["gpu_device"].as<int>(INVALID_GPU_ID);
  rmax_rx_config.max_packet_size = rmax_rx_settings["max_packet_size"].as<uint16_t>(1500);
  rmax_rx_config.num_concurrent_batches =
      rmax_rx_settings["num_concurrent_batches"].as<uint32_t>(10);
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
AdvNetStatus RmaxConfigManager::parse_tx_queue_rivermax_config(const YAML::Node& q_item,
                                                               TxQueueConfig& q) {
  return AdvNetStatus::SUCCESS;
}

};  // namespace holoscan::ops
