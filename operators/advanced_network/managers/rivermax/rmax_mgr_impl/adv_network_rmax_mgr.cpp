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

#include <atomic>
#include <cmath>
#include <complex>
#include <chrono>
#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <sys/time.h>
#include <vector>
#include <tuple>
#include <cassert>

#include "adv_network_rmax_mgr.h"
#include "rmax_mgr_impl/burst_manager.h"
#include "rmax_mgr_impl/packet_processor.h"
#include "rmax_mgr_impl/rmax_chunk_consumer_ano.h"
#include "rmax_ipo_receiver_service.h"
#include <holoscan/logger/logger.hpp>
#include "rt_threads.h"

using namespace std::chrono;

namespace holoscan::ops {

using namespace ral::services::rmax_ipo_receiver;

static constexpr int USECS_IN_SECOND = 1000000;

/**
 * @brief Configuration structure for Rmax RX queue.
 *
 * This structure holds the configuration settings for an Rmax RX queue,
 * including packet size, chunk size, IP addresses, ports, and other parameters.
 */
struct RmaxRxQueueConfig : public AnoMgrExtraQueueConfig {
  uint16_t max_packet_size = 0;
  size_t max_chunk_size;
  uint32_t num_concurrent_batches;
  bool gpu_direct;
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
  ~RmaxRxQueueConfig() = default;
};

struct ExtRmaxIPOReceiverConfig : RmaxIPOReceiverConfig {
  bool send_packet_ext_info;
};

class RmaxConfigManager {
 public:
  static constexpr uint16_t RMAX_DEFAULT_LOG_LEVEL = 3;
  static constexpr uint16_t RMAX_MIN_LOG_LEVEL = 1;
  static constexpr uint16_t RMAX_MAX_LOG_LEVEL = 6;

  bool parse_configuration(const AdvNetConfigYaml& cfg, std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr);
  uint16_t get_rmax_log_level() const { return rmax_log_level; }
  
  const std::unordered_map<uint32_t, ExtRmaxIPOReceiverConfig>& get_rx_service_configs() const {
    return rx_service_configs;
  }  

 private:
  void set_default_config(ExtRmaxIPOReceiverConfig& rx_service_cfg);
  bool configure_rx_queue(uint16_t port_id, int master_core, int gpu_id, const RxQueueConfig& q);
  bool build_rmax_ipo_receiver_config(const RmaxRxQueueConfig& rmax_rx_config, int master_core,
                                      const std::string& cores, int gpu_id, int split_boundary,
                                      ExtRmaxIPOReceiverConfig& rx_service_cfg);
  bool validate_rx_queue_config(const RmaxRxQueueConfig& rmax_rx_config);
  void set_rx_service_common_app_settings(AppSettings& app_settings_config,
                                          const RmaxRxQueueConfig& rmax_rx_config, int master_core,
                                          int gpu_id, int split_boundary);
  void set_allocator_type(AppSettings& app_settings_config, const std::string& allocator_type);
  bool parse_and_set_cores(AppSettings& app_settings_config, const std::string& cores);
  void set_rx_service_ipo_receiver_settings(ExtRmaxIPOReceiverConfig& rx_service_cfg,
                                            const RmaxRxQueueConfig& rmax_rx_config);
  void add_new_rx_service_config(ExtRmaxIPOReceiverConfig& rx_service_cfg, uint16_t port_id,
                                 uint16_t queue_id);

  uint16_t rmax_log_level = RMAX_DEFAULT_LOG_LEVEL;
  std::unordered_map<uint32_t, ExtRmaxIPOReceiverConfig> rx_service_configs;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr;
};



/**
 * @brief Implementation class for RmaxMgr.
 *
 * This class contains the implementation details for RmaxMgr, including
 * methods for configuration, initialization, packet handling, and statistics.
 */
class RmaxMgr::RmaxMgrImpl {
 public:
  RmaxMgrImpl() = default;
  ~RmaxMgrImpl();

  bool set_config_and_initialize(const AdvNetConfigYaml& cfg);
  void initialize();
  void run();

  void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx);
  void* get_pkt_ptr(AdvNetBurstParams* burst, int idx);
  uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx);
  uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx);
  void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx);
  AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams* burst);
  AdvNetStatus set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr);
  AdvNetStatus set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                            unsigned int src_host, unsigned int dst_host);
  AdvNetStatus set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                           uint16_t dst_port);
  AdvNetStatus set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len);
  bool tx_burst_available(AdvNetBurstParams* burst);

  AdvNetStatus set_pkt_lens(AdvNetBurstParams* burst, int idx,
                            const std::initializer_list<int>& lens);
  void free_all_seg_pkts(AdvNetBurstParams* burst, int seg);
  void free_all_pkts(AdvNetBurstParams* burst);
  void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt);
  void free_pkt(AdvNetBurstParams* burst, int pkt);
  void free_rx_burst(AdvNetBurstParams* burst);
  void free_tx_burst(AdvNetBurstParams* burst);
  void format_eth_addr(char* dst, std::string addr);
  std::optional<uint16_t> get_port_from_ifname(const std::string& name);

  AdvNetStatus get_rx_burst(AdvNetBurstParams** burst);
  AdvNetStatus set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_meta(AdvNetBurstParams* burst);
  void free_tx_meta(AdvNetBurstParams* burst);
  AdvNetStatus get_tx_meta_buf(AdvNetBurstParams** burst);
  AdvNetStatus send_tx_burst(AdvNetBurstParams* burst);
  void shutdown();
  void print_stats();
  uint64_t get_burst_tot_byte(AdvNetBurstParams* burst);
  AdvNetBurstParams* create_burst_params();
  AdvNetStatus get_mac(int port, char* mac);
  int address_to_port(const std::string& addr);

 private:
  static constexpr double GIGABYTE = 1073741824.0;
  static constexpr double MEGABYTE = 1048576.0;
    static constexpr int DEFAULT_NUM_RX_BURST = 64;

  static void flush_packets(int port);
  void setup_accurate_send_scheduling_mask();
  int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
  void initialize_rx_service(uint32_t service_id, const ExtRmaxIPOReceiverConfig& config);
  void print_total_stats();
  void print_stream_stats(std::stringstream& ss, uint32_t stream_index,
                          IPORXStatistics stream_stats,
                          std::vector<IPOPathStatistics> stream_path_stats);
  AdvNetConfigYaml cfg_;
  std::unordered_map<uint32_t,
                     std::unique_ptr<ral::services::rmax_ipo_receiver::RmaxIPOReceiverService>>
      rx_services;
  std::unordered_map<uint32_t, std::unique_ptr<RmaxChunkConsumerAno>> rmax_chunk_consumers;
  std::unordered_map<uint32_t, std::shared_ptr<RxBurstsManager>> rx_burst_managers;
  std::unordered_map<uint32_t, std::shared_ptr<RxPacketProcessor>> rx_packet_processors;
  std::shared_ptr<AnoBurstsQueue> rx_bursts_out_queue;
  std::vector<std::thread> rx_service_threads;
  bool initialized_ = false;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr;
  // Instances of the new classes
  RmaxConfigManager config_manager;  
};

std::atomic<bool> force_quit = false;

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
 * @return True if the configuration was successfully parsed, false otherwise.
 */
bool RmaxConfigManager::parse_configuration(const AdvNetConfigYaml& cfg, std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib) {
  int rmax_rx_config_found = 0;
  
  this->rmax_apps_lib = rmax_apps_lib;
  
  for (const auto& intf : cfg.ifs_) {
    HOLOSCAN_LOG_INFO("Rmax init Port {} -- RX: {} TX: {}",
                      intf.port_id_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");

    for (const auto& q : intf.rx_.queues_) {
      if (!configure_rx_queue(intf.port_id_,
                              cfg.common_.master_core_,
                              cfg.mrs_.at(q.common_.mrs_[0]).affinity_,
                              q)) {
        continue;
      }
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
 * @param gpu_id The GPU ID.
 * @param q The RX queue configuration.
 * @return True if the configuration is successful, false otherwise.
 */
bool RmaxConfigManager::configure_rx_queue(uint16_t port_id, int master_core, int gpu_id,
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

  auto& rmax_rx_config = *rmax_rx_config_ptr;
  if (!validate_rx_queue_config(rmax_rx_config)) { return false; }

  ExtRmaxIPOReceiverConfig rx_service_cfg;

  if (!build_rmax_ipo_receiver_config(rmax_rx_config,
                                     master_core,
                                     q.common_.cpu_core_,
                                     gpu_id,
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
 * @param gpu_id The GPU ID.
 * @param split_boundary The split boundary value.
 * @param rx_service_cfg The RX service configuration to be built.
 * @return True if the configuration is successful, false otherwise.
 */
bool RmaxConfigManager::build_rmax_ipo_receiver_config(const RmaxRxQueueConfig& rmax_rx_config,
                                                         int master_core, const std::string& cores,
                                                         int gpu_id, int split_boundary,
                                                         ExtRmaxIPOReceiverConfig& rx_service_cfg) {
  rx_service_cfg.app_settings = std::make_shared<AppSettings>();
  set_default_config(rx_service_cfg);

  auto& app_settings_config = *(rx_service_cfg.app_settings);
  set_rx_service_common_app_settings(
      app_settings_config, rmax_rx_config, master_core, gpu_id, split_boundary);

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
 * @param gpu_id The GPU ID.
 * @param split_boundary The split boundary value.
 */
void RmaxConfigManager::set_rx_service_common_app_settings(
    AppSettings& app_settings_config, const RmaxRxQueueConfig& rmax_rx_config, int master_core,
    int gpu_id, int split_boundary) {
  app_settings_config.local_ips = rmax_rx_config.local_ips;
  app_settings_config.source_ips = rmax_rx_config.source_ips;
  app_settings_config.destination_ips = rmax_rx_config.destination_ips;
  app_settings_config.destination_ports = rmax_rx_config.destination_ports;

  if (rmax_rx_config.gpu_direct) {
    app_settings_config.gpu_id = gpu_id;
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
 * @brief Adds a new RX service configuration to configuration map.
 *
 * @param rx_service_cfg The RX service configuration.
 * @param port_id The port ID.
 * @param queue_id The queue ID.
 */
void RmaxConfigManager::add_new_rx_service_config(ExtRmaxIPOReceiverConfig& rx_service_cfg,
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
 * @brief Init
 *
 * This function sets the configuration and initializes the Rmax manager.
 * It starts the initialization in a separate thread to avoid setting the
 * affinity for the whole application. Once initialized, it runs the manager.
 *
 * @param cfg The configuration YAML.
 * @return True if the initialization was successful, false otherwise.
 */
bool RmaxMgr::RmaxMgrImpl::set_config_and_initialize(const AdvNetConfigYaml& cfg) {
  if (!this->initialized_) {
    cfg_ = cfg;

    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&RmaxMgr::RmaxMgrImpl::initialize, this);
    t.join();

    this->initialized_ = true;
    run();
  }
  return true;
}

/**
 * @brief Initializes the Rmax manager.
 *
 * This function initializes the Rmax manager by setting up the RX bursts queue,
 * creating the Rmax applications library facade, parsing the configuration,
 * and initializing the RX services and chunk consumers.
 */
void RmaxMgr::RmaxMgrImpl::initialize() {
  // Create the RX bursts output queue
  rx_bursts_out_queue = std::make_shared<AnoBurstsQueue>();

  // Create the Rmax applications library facade
  rmax_apps_lib = std::make_shared<ral::lib::RmaxAppsLibFacade>();

  // Parse the configuration
  bool res = config_manager.parse_configuration(cfg_, rmax_apps_lib);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse configuration for RMAX ANO Manager");
    return;
  }

  rivermax_setparam("RIVERMAX_LOG_LEVEL", std::to_string(config_manager.get_rmax_log_level()), true);

  // Iterate over the RX service configurations and initialize each RX service
  const auto& rx_service_configs = config_manager.get_rx_service_configs();
  for (const auto& entry : rx_service_configs) { initialize_rx_service(entry.first, entry.second); }
}

/**
 * @brief Initializes an RX service with the given configuration.
 *
 * This method creates and initializes an RX service based on the provided configuration.
 * It also sets up the necessary burst managers, packet processors, and chunk consumers.
 *
 * @param service_id The unique service id identifying the RX service.
 * @param config The configuration for the RX service.
 */
void RmaxMgr::RmaxMgrImpl::initialize_rx_service(uint32_t service_id,
                                                 const ExtRmaxIPOReceiverConfig& config) {
  uint16_t port_id = RmaxBurst::burst_port_id_from_burst_tag(service_id);
  uint16_t queue_id = RmaxBurst::burst_queue_id_from_burst_tag(service_id);

  // Create and initialize the RX service
  auto rx_service = std::make_unique<RmaxIPOReceiverService>(config);
  auto init_status = rx_service->get_init_status();
  if (init_status != ReturnStatus::obj_init_success) {
    HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
    return;
  }

  // Store the RX service and create the chunk consumer
  rx_services[service_id] = std::move(rx_service);
  rx_burst_managers[service_id] = std::make_shared<RxBurstsManager>(config.send_packet_ext_info,
                                                                    port_id,
                                                                    queue_id,
                                                                    config.max_chunk_size,
                                                                    config.app_settings->gpu_id,
                                                                    rx_bursts_out_queue);

  rx_packet_processors[service_id] =
      std::make_shared<RxPacketProcessor>(rx_burst_managers[service_id]);

  rmax_chunk_consumers[service_id] =
      std::make_unique<RmaxChunkConsumerAno>(rx_packet_processors[service_id]);

  // Set the chunk consumer for the RX service
  rx_services[service_id]->set_chunk_consumer(rmax_chunk_consumers[service_id].get());
}

void RmaxMgr::RmaxMgrImpl::print_stream_stats(std::stringstream& ss, uint32_t stream_index,
                                              IPORXStatistics stream_stats,
                                              std::vector<IPOPathStatistics> stream_path_stats) {
  ss << "[stream_index " << std::setw(3) << stream_index << "]"
     << " Got " << std::setw(7) << stream_stats.rx_counter << " packets | ";

  if (stream_stats.received_bytes >= GIGABYTE) {
    // Display in gigabytes
    ss << std::fixed << std::setprecision(2) << (stream_stats.received_bytes / GIGABYTE) << " GB |";
  } else if (stream_stats.received_bytes >= MEGABYTE) {
    // Display in megabytes
    ss << std::fixed << std::setprecision(2) << (stream_stats.received_bytes / MEGABYTE) << " MB |";
  } else {
    // Display in bytes
    ss << stream_stats.received_bytes << " bytes |";
  }

  ss << " dropped: ";
  for (uint32_t s_index = 0; s_index < stream_path_stats.size(); ++s_index) {
    if (s_index > 0) { ss << ", "; }
    ss << stream_path_stats[s_index].rx_dropped + stream_stats.rx_dropped;
  }
  ss << " |"
     << " consumed: " << stream_stats.consumed_packets << " |"
     << " unconsumed: " << stream_stats.unconsumed_packets << " |"
     << " lost: " << stream_stats.rx_dropped << " |"
     << " exceed MD: " << stream_stats.rx_exceed_md << " |"
     << " bad RTP hdr: " << stream_stats.rx_corrupt_rtp_header << " | ";

  for (uint32_t s_index = 0; s_index < stream_path_stats.size(); ++s_index) {
    if (stream_stats.rx_counter > 0) {
      uint32_t number = static_cast<uint32_t>(
          floor(100 * static_cast<double>(stream_path_stats[s_index].rx_count) /
                static_cast<double>(stream_stats.rx_counter)));
      ss << " " << std::setw(3) << number << "%";
    } else {
      ss << "   0%";
    }
  }
  ss << "\n";
}

/**
 * @brief Prints the statistics of the Rmax manager.
 */
void RmaxMgr::RmaxMgrImpl::print_total_stats() {
  // Implementation for printing statistics
  std::stringstream ss;
  uint32_t stream_id = 0;
  ss << "RMAX ANO Statistics\n";
  ss << "====================\n";
  ss << "Total Statistics\n";
  ss << "----------------\n";
  for (const auto& entry : rx_services) {
    uint32_t key = entry.first;
    auto& rx_service = entry.second;
    auto [stream_stats, path_stats] = rx_service->get_streams_statistics();
    for (uint32_t i = 0; i < stream_stats.size(); ++i) {
      print_stream_stats(ss, stream_id++, stream_stats[i], path_stats[i]);
    }
  }
  HOLOSCAN_LOG_INFO(ss.str());
}

/**
 * @brief Destructor for RmaxMgrImpl.
 *
 * This destructor prints the statistics and joins all RX service threads
 * before destroying the Rmax manager implementation.
 */
RmaxMgr::RmaxMgrImpl::~RmaxMgrImpl() {
  print_total_stats();
  for (auto& rx_service_thread : rx_service_threads) {
    if (rx_service_thread.joinable()) { rx_service_thread.join(); }
  }
}

/**
 * @brief Runs the Rmax manager.
 *
 * This function starts the RX services by iterating over the RX services map
 * and launching each service in a separate thread. It ensures that each service
 * is properly initialized before running.
 */
void RmaxMgr::RmaxMgrImpl::run() {
  HOLOSCAN_LOG_INFO("Starting RX Services");
  for (const auto& entry : rx_services) {
    uint32_t key = entry.first;
    auto& rx_service = entry.second;
    if (rx_service->get_init_status() != ReturnStatus::obj_init_success) {
      HOLOSCAN_LOG_ERROR("Rx Service failed to initialize, cannot run");
      return;
    }
    rx_service_threads.emplace_back([rx_service_ptr = rx_service.get()]() {
      ReturnStatus status = rx_service_ptr->run();
      // Handle the status if needed
    });
    // Use a lambda to capture the raw pointer and invoke the run method
  }
  HOLOSCAN_LOG_INFO("Done starting workers");
}

/**
 * @brief Flushes packets on a specific port.
 *
 * @param port The port number on which to flush packets.
 */
void RmaxMgr::RmaxMgrImpl::flush_packets(int port) {
  HOLOSCAN_LOG_INFO("Flushing packet on port {}", port);
}

/**
 * @brief Gets the pointer to a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param idx The packet index within the segment.
 * @return Pointer to the packet.
 */
void* RmaxMgr::RmaxMgrImpl::get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
  return burst->pkts[seg][idx];
}

/**
 * @brief Gets the pointer to a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Pointer to the packet.
 */
void* RmaxMgr::RmaxMgrImpl::get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  return burst->pkts[0][idx];
}

/**
 * @brief Gets the length of a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param idx The packet index within the segment.
 * @return Length of the packet.
 */
uint16_t RmaxMgr::RmaxMgrImpl::get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
  return burst->pkt_lens[seg][idx];
}

/**
 * @brief Gets the length of a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Length of the packet.
 */
uint16_t RmaxMgr::RmaxMgrImpl::get_pkt_len(AdvNetBurstParams* burst, int idx) {
  return burst->pkt_lens[0][idx];
}

/**
 * @brief Gets the extra information of a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Pointer to the extra information.
 */
void* RmaxMgr::RmaxMgrImpl::get_pkt_extra_info(AdvNetBurstParams* burst, int idx) {
  RmaxBurst* rmax_burst = static_cast<RmaxBurst*>(burst);
  auto burst_flags = rmax_burst->get_burst_flags();
  if (rmax_burst->is_packet_info_per_packet()) return burst->pkt_extra_info[idx];
  return nullptr;
}

/**
 * @brief Sets the transmission time for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param timestamp The transmission timestamp.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_pkt_tx_time(AdvNetBurstParams* burst, int idx,
                                                   uint64_t timestamp) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Gets a burst of TX packets.
 *
 * @param burst The burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::get_tx_pkt_burst(AdvNetBurstParams* burst) {
  return AdvNetStatus::NO_FREE_BURST_BUFFERS;
}

/**
 * @brief Sets the Ethernet header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param dst_addr The destination address.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Sets the IPv4 header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param ip_len The length of the IP packet.
 * @param proto The protocol.
 * @param src_host The source host address.
 * @param dst_host The destination host address.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len,
                                                uint8_t proto, unsigned int src_host,
                                                unsigned int dst_host) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Sets the UDP header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param udp_len The length of the UDP packet.
 * @param src_port The source port.
 * @param dst_port The destination port.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len,
                                               uint16_t src_port, uint16_t dst_port) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Sets the UDP payload for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param data The payload data.
 * @param len The length of the payload data.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_udp_payload(AdvNetBurstParams* burst, int idx, void* data,
                                                   int len) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Checks if a TX burst is available.
 *
 * @param burst The burst parameters.
 * @return True if a TX burst is available, false otherwise.
 */
bool RmaxMgr::RmaxMgrImpl::tx_burst_available(AdvNetBurstParams* burst) {
  return false;
}

/**
 * @brief Sets the packet lengths for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param lens The list of lengths.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                                const std::initializer_list<int>& lens) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Frees all packets in a specific segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 */
void RmaxMgr::RmaxMgrImpl::free_all_seg_pkts(AdvNetBurstParams* burst, int seg) {}

/**
 * @brief Frees all packets.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::RmaxMgrImpl::free_all_pkts(AdvNetBurstParams* burst) {}

/**
 * @brief Frees a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param pkt The packet index within the segment.
 */
void RmaxMgr::RmaxMgrImpl::free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) {}

/**
 * @brief Frees a specific packet.
 *
 * @param burst The burst parameters.
 * @param pkt The packet index.
 */
void RmaxMgr::RmaxMgrImpl::free_pkt(AdvNetBurstParams* burst, int pkt) {}

/**
 * @brief Frees the RX burst.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::RmaxMgrImpl::free_rx_burst(AdvNetBurstParams* burst) {
  uint32_t key =
      RmaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  if (rx_services.find(key) == rx_services.end() ||
      rx_burst_managers.find(key) == rx_burst_managers.end()) {
    HOLOSCAN_LOG_ERROR("Rmax Service is not initialized");
    return;
  }

  auto& rx_service = rx_services[key];
  auto& rmax_bursts_manager = rx_burst_managers[key];

  if (!(rx_service->is_alive())) {
    HOLOSCAN_LOG_ERROR("Rmax Service is not initialized");
    return;
  }

  rmax_bursts_manager->rx_burst_done(static_cast<RmaxBurst*>(burst));
}

/**
 * @brief Frees the TX burst.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::RmaxMgrImpl::free_tx_burst(AdvNetBurstParams* burst) {}

/**
 * @brief Gets the port number from an interface name.
 *
 * @param name The interface name.
 * @return Optional containing the port number if found, otherwise empty.
 */
std::optional<uint16_t> RmaxMgr::RmaxMgrImpl::get_port_from_ifname(const std::string& name) {
  HOLOSCAN_LOG_INFO("Port name {}", name);
  for (const auto& intf : cfg_.ifs_) {
    if (name == intf.address_) { return intf.port_id_; }
  }
  return -1;
}

/**
 * @brief Dequeues an RX burst.
 *
 * @param burst Pointer to the burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::get_rx_burst(AdvNetBurstParams** burst) {
  auto out_burst = rx_bursts_out_queue->dequeue_burst().get();
  *burst = static_cast<AdvNetBurstParams*>(out_burst);
  if (*burst == nullptr) { return AdvNetStatus::NOT_READY; }
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Frees the RX metadata.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::RmaxMgrImpl::free_rx_meta(AdvNetBurstParams* burst) {}

/**
 * @brief Frees the TX metadata.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::RmaxMgrImpl::free_tx_meta(AdvNetBurstParams* burst) {}

/**
 * @brief Gets the TX metadata buffer.
 *
 * @param burst Pointer to the burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::get_tx_meta_buf(AdvNetBurstParams** burst) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Sends a TX burst.
 *
 * @param burst The burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::send_tx_burst(AdvNetBurstParams* burst) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Shuts down the Rmax manager.
 */
void RmaxMgr::RmaxMgrImpl::shutdown() {
  if (!force_quit.load()) {
    HOLOSCAN_LOG_INFO("ANO RMAX manager shutting down");
    // Nodes are waiting to signals to exit
    std::raise(SIGINT);
    force_quit.store(false);
  }
}

/**
 * @brief Prints the statistics of the Rmax manager.
 */
void RmaxMgr::RmaxMgrImpl::print_stats() {
  print_total_stats();
}

/**
 * @brief Gets the total byte count of a burst.
 *
 * @param burst The burst parameters.
 * @return Total byte count of the burst.
 */
uint64_t RmaxMgr::RmaxMgrImpl::get_burst_tot_byte(AdvNetBurstParams* burst) {
  return 0;
}

/**
 * @brief Creates burst parameters.
 *
 * @return Pointer to the created burst parameters.
 */
AdvNetBurstParams* RmaxMgr::RmaxMgrImpl::create_burst_params() {
  return new AdvNetBurstParams();
}

/**
 * @brief Gets the MAC address for a specific port.
 *
 * @param port The port number.
 * @param mac Pointer to the MAC address buffer.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::RmaxMgrImpl::get_mac(int port, char* mac) {
  return AdvNetStatus::NOT_SUPPORTED;
}

/**
 * @brief Converts an address to a port number.
 *
 * @param addr The address.
 * @return The port number.
 */
int RmaxMgr::RmaxMgrImpl::address_to_port(const std::string& addr) {
  for (const auto& intf : cfg_.ifs_) {
    if (intf.address_ == addr) { return intf.port_id_; }
  }
  return -1;
}

/**
 * @brief Constructor for RmaxMgr.
 */
RmaxMgr::RmaxMgr() : pImpl(std::make_unique<RmaxMgrImpl>()) {}

/**
 * @brief Destructor for RmaxMgr.
 */
RmaxMgr::~RmaxMgr() = default;

/**
 * @brief Sets the configuration and initializes the Rmax manager.
 *
 * @param cfg The configuration YAML.
 * @return True if the initialization was successful, false otherwise.
 */
bool RmaxMgr::set_config_and_initialize(const AdvNetConfigYaml& cfg) {
  bool res = true;
  if (!this->initialized_) {
    res = pImpl->set_config_and_initialize(cfg);
    this->initialized_ = res;
  }
  return res;
}

/**
 * @brief Initializes the Rmax manager.
 */
void RmaxMgr::initialize() {
  pImpl->initialize();
}

/**
 * @brief Runs the Rmax manager.
 */
void RmaxMgr::run() {
  pImpl->run();
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
AdvNetStatus RmaxMgr::parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q) {
  // Extract the rmax_rx_settings node from the YAML configuration
  const auto& rmax_rx_settings = q_item["rmax_rx_settings"];

  // Check if rmax_rx_settings node exists
  if (!rmax_rx_settings) {
    HOLOSCAN_LOG_ERROR("Rmax RX settings not found");
    return AdvNetStatus::INVALID_PARAMETER;
  }

  // Allocate memory for extra queue configuration
  q.common_.extra_queue_config_ = new RmaxRxQueueConfig();
  auto& rmax_rx_config = *(reinterpret_cast<RmaxRxQueueConfig*>(q.common_.extra_queue_config_));

  // Parse local IP addresses
  for (const auto& q_item_ip : rmax_rx_settings["local_ip_addresses"]) {
    rmax_rx_config.local_ips.emplace_back(q_item_ip.as<std::string>());
  }

  // Parse source IP addresses
  for (const auto& q_item_ip : rmax_rx_settings["source_ip_addresses"]) {
    rmax_rx_config.source_ips.emplace_back(q_item_ip.as<std::string>());
  }

  // Parse destination IP addresses
  for (const auto& q_item_ip : rmax_rx_settings["destination_ip_addresses"]) {
    rmax_rx_config.destination_ips.emplace_back(q_item_ip.as<std::string>());
  }

  // Parse destination ports
  for (const auto& q_item_ip : rmax_rx_settings["destination_ports"]) {
    rmax_rx_config.destination_ports.emplace_back(q_item_ip.as<uint16_t>());
  }

  // Parse additional RX queue settings
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
AdvNetStatus RmaxMgr::parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q) {
  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Gets the pointer to a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param idx The packet index within the segment.
 * @return Pointer to the packet.
 */
void* RmaxMgr::get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
  return pImpl->get_seg_pkt_ptr(burst, seg, idx);
}

/**
 * @brief Gets the pointer to a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Pointer to the packet.
 */
void* RmaxMgr::get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  return pImpl->get_pkt_ptr(burst, idx);
}

/**
 * @brief Gets the length of a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param idx The packet index within the segment.
 * @return Length of the packet.
 */
uint16_t RmaxMgr::get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
  return pImpl->get_seg_pkt_len(burst, seg, idx);
}

/**
 * @brief Gets the length of a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Length of the packet.
 */
uint16_t RmaxMgr::get_pkt_len(AdvNetBurstParams* burst, int idx) {
  return pImpl->get_pkt_len(burst, idx);
}

/**
 * @brief Gets the extra information of a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Pointer to the extra information.
 */
void* RmaxMgr::get_pkt_extra_info(AdvNetBurstParams* burst, int idx) {
  return pImpl->get_pkt_extra_info(burst, idx);
}

/**
 * @brief Gets a burst of TX packets.
 *
 * @param burst The burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::get_tx_pkt_burst(AdvNetBurstParams* burst) {
  return pImpl->get_tx_pkt_burst(burst);
}

/**
 * @brief Sets the Ethernet header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param dst_addr The destination address.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) {
  return pImpl->set_eth_hdr(burst, idx, dst_addr);
}

/**
 * @brief Sets the IPv4 header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param ip_len The length of the IP packet.
 * @param proto The protocol.
 * @param src_host The source host address.
 * @param dst_host The destination host address.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                   unsigned int src_host, unsigned int dst_host) {
  return pImpl->set_ipv4_hdr(burst, idx, ip_len, proto, src_host, dst_host);
}

/**
 * @brief Sets the UDP header for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param udp_len The length of the UDP packet.
 * @param src_port The source port.
 * @param dst_port The destination port.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                  uint16_t dst_port) {
  return pImpl->set_udp_hdr(burst, idx, udp_len, src_port, dst_port);
}

/**
 * @brief Sets the UDP payload for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param data The payload data.
 * @param len The length of the payload data.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) {
  return pImpl->set_udp_payload(burst, idx, data, len);
}

/**
 * @brief Checks if a TX burst is available.
 *
 * @param burst The burst parameters.
 * @return True if a TX burst is available, false otherwise.
 */
bool RmaxMgr::tx_burst_available(AdvNetBurstParams* burst) {
  return pImpl->tx_burst_available(burst);
}

/**
 * @brief Sets the packet lengths for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param lens The list of lengths.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                   const std::initializer_list<int>& lens) {
  return pImpl->set_pkt_lens(burst, idx, lens);
}

/**
 * @brief Frees all packets in a specific segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 */
void RmaxMgr::free_all_seg_pkts(AdvNetBurstParams* burst, int seg) {
  pImpl->free_all_seg_pkts(burst, seg);
}

/**
 * @brief Frees all packets.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::free_all_pkts(AdvNetBurstParams* burst) {
  pImpl->free_all_pkts(burst);
}

/**
 * @brief Frees a specific packet in a segment.
 *
 * @param burst The burst parameters.
 * @param seg The segment index.
 * @param pkt The packet index within the segment.
 */
void RmaxMgr::free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) {
  pImpl->free_pkt_seg(burst, seg, pkt);
}

/**
 * @brief Frees a specific packet.
 *
 * @param burst The burst parameters.
 * @param pkt The packet index.
 */
void RmaxMgr::free_pkt(AdvNetBurstParams* burst, int pkt) {
  pImpl->free_pkt(burst, pkt);
}

/**
 * @brief Frees the RX burst.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::free_rx_burst(AdvNetBurstParams* burst) {
  pImpl->free_rx_burst(burst);
}

/**
 * @brief Frees the TX burst.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::free_tx_burst(AdvNetBurstParams* burst) {
  pImpl->free_tx_burst(burst);
}

/**
 * @brief Gets the port number from an interface name.
 *
 * @param name The interface name.
 * @return Optional containing the port number if found, otherwise empty.
 */
std::optional<uint16_t> RmaxMgr::get_port_from_ifname(const std::string& name) {
  return pImpl->get_port_from_ifname(name);
}

/**
 * @brief Gets an RX burst.
 *
 * @param burst Pointer to the burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::get_rx_burst(AdvNetBurstParams** burst) {
  return pImpl->get_rx_burst(burst);
}

/**
 * @brief Sets the transmission time for a specific packet.
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @param timestamp The transmission timestamp.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp) {
  return pImpl->set_pkt_tx_time(burst, idx, timestamp);
}

/**
 * @brief Frees the RX metadata.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::free_rx_meta(AdvNetBurstParams* burst) {
  pImpl->free_rx_meta(burst);
}

/**
 * @brief Frees the TX metadata.
 *
 * @param burst The burst parameters.
 */
void RmaxMgr::free_tx_meta(AdvNetBurstParams* burst) {
  pImpl->free_tx_meta(burst);
}

/**
 * @brief Gets the TX metadata buffer.
 *
 * @param burst Pointer to the burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::get_tx_meta_buf(AdvNetBurstParams** burst) {
  return pImpl->get_tx_meta_buf(burst);
}

/**
 * @brief Sends a TX burst.
 *
 * @param burst The burst parameters.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::send_tx_burst(AdvNetBurstParams* burst) {
  return pImpl->send_tx_burst(burst);
}

/**
 * @brief Shuts down the Rmax manager.
 */
void RmaxMgr::shutdown() {
  pImpl->shutdown();
}

/**
 * @brief Prints the statistics of the Rmax manager.
 */
void RmaxMgr::print_stats() {
  pImpl->print_stats();
}

/**
 * @brief Gets the total byte count of a burst.
 *
 * @param burst The burst parameters.
 * @return Total byte count of the burst.
 */
uint64_t RmaxMgr::get_burst_tot_byte(AdvNetBurstParams* burst) {
  return pImpl->get_burst_tot_byte(burst);
}

/**
 * @brief Creates burst parameters.
 *
 * @return Pointer to the created burst parameters.
 */
AdvNetBurstParams* RmaxMgr::create_burst_params() {
  return pImpl->create_burst_params();
}

/**
 * @brief Gets the MAC address for a specific port.
 *
 * @param port The port number.
 * @param mac Pointer to the MAC address buffer.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::get_mac(int port, char* mac) {
  return pImpl->get_mac(port, mac);
}

/**
 * @brief Converts an address to a port number.
 *
 * @param addr The address.
 * @return The port number.
 */
int RmaxMgr::address_to_port(const std::string& addr) {
  return pImpl->address_to_port(addr);
}

};  // namespace holoscan::ops
