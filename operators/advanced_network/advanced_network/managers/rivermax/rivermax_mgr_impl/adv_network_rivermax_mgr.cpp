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

#include <atomic>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <sys/time.h>
#include <vector>
#include <tuple>
#include <cassert>

#include <holoscan/logger/logger.hpp>

#include "rt_threads.h"
#include "rdk/rivermax_dev_kit.h"

#include "adv_network_rivermax_mgr.h"
#include "rivermax_mgr_impl/rivermax_config_manager.h"
#include "rivermax_mgr_impl/burst_manager.h"
#include "rivermax_mgr_impl/packet_processor.h"
#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"
#include "rivermax_mgr_impl/rivermax_mgr_service.h"

using namespace std::chrono;

namespace holoscan::advanced_network {

/**
 * A map of log level to a tuple of the description and command strings.
 */
const std::unordered_map<RivermaxLogLevel::Level, std::tuple<std::string, std::string>>
    RivermaxLogLevel::level_to_cmd_map = {
        {TRACE, {"Trace", "0"}},
        {DEBUG, {"Debug", "1"}},
        {INFO, {"Info", "2"}},
        {WARN, {"Warning", "3"}},
        {ERROR, {"Error", "4"}},
        {CRITICAL, {"Critical", "5"}},
        {OFF, {"Disabled", "6"}},
};

/**
 * A map of advanced_network log level to Rivermax log level.
 */
const std::unordered_map<LogLevel::Level, RivermaxLogLevel::Level>
    RivermaxLogLevel::adv_net_to_rivermax_log_level_map = {
        {LogLevel::TRACE, TRACE},
        {LogLevel::DEBUG, DEBUG},
        {LogLevel::INFO, INFO},
        {LogLevel::WARN, WARN},
        {LogLevel::ERROR, ERROR},
        {LogLevel::CRITICAL, CRITICAL},
        {LogLevel::OFF, OFF},
};

/**
 * @brief Implementation class for RivermaxMgr.
 *
 * This class contains the implementation details for RivermaxMgr, including
 * methods for configuration, initialization, packet handling, and statistics.
 */
class RivermaxMgr::RivermaxMgrImpl {
 public:
  RivermaxMgrImpl() = default;
  ~RivermaxMgrImpl();

  bool set_config_and_initialize(const NetworkConfig& cfg);
  void initialize();
  void run();

  void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx);
  void* get_packet_ptr(BurstParams* burst, int idx);
  uint16_t get_segment_packet_length(BurstParams* burst, int seg, int idx);
  uint16_t get_packet_length(BurstParams* burst, int idx);
  void* get_packet_extra_info(BurstParams* burst, int idx);
  Status get_tx_packet_burst(BurstParams* burst);
  Status set_eth_header(BurstParams* burst, int idx, char* dst_addr);
  Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                         unsigned int src_host, unsigned int dst_host);
  Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                        uint16_t dst_port);
  Status set_udp_payload(BurstParams* burst, int idx, void* data, int len);
  bool is_tx_burst_available(BurstParams* burst);

  Status set_packet_lengths(BurstParams* burst, int idx, const std::initializer_list<int>& lens);
  void free_all_segment_packets(BurstParams* burst, int seg);
  void free_all_packets(BurstParams* burst);
  void free_packet_segment(BurstParams* burst, int seg, int pkt);
  void free_packet(BurstParams* burst, int pkt);
  void free_rx_burst(BurstParams* burst);
  void free_tx_burst(BurstParams* burst);
  void format_eth_addr(char* dst, std::string addr);
  Status get_rx_burst(BurstParams** burst, int port, int q, int stream_id);
  Status get_rx_burst(BurstParams** burst, int port, int q);
  Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_metadata(BurstParams* burst);
  void free_tx_metadata(BurstParams* burst);
  Status get_tx_metadata_buffer(BurstParams** burst);
  Status send_tx_burst(BurstParams* burst);
  void shutdown();
  void print_stats();
  uint64_t get_burst_tot_byte(BurstParams* burst);
  BurstParams* create_tx_burst_params();
  Status get_mac_addr(int port, char* mac);

 private:
  template<typename ConfigType>
    ConfigType* find_rivermax_config(uint32_t service_id, QueueConfigType expected_type);
  template<typename ConfigType>
    std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>>
    create_burst_queues_for_streams(const ConfigType& rivermax_config, uint32_t service_id);
  template<typename ConfigType, typename BuilderType, typename ServiceType>
    bool common_initialize_rx_service(uint32_t service_id,
                                              std::shared_ptr<ConfigBuilderHolder> config_holder,
                                              QueueConfigType expected_type,
                                              const std::string& service_name);
  bool initialize_rx_service(uint32_t service_id,
                             std::shared_ptr<ConfigBuilderHolder> config_holder);
  bool initialize_tx_service(uint32_t service_id,
                             std::shared_ptr<ConfigBuilderHolder> config_holder);

 private:
  static constexpr int DEFAULT_NUM_RX_BURST = 64;
  static constexpr int MAX_TX_BURST = 128;
  static constexpr int MAX_NUM_OF_FRAMES_IN_BURST = 1;

  const NetworkConfig* cfg_ = nullptr;
  std::unordered_map<uint32_t, std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>>> rx_bursts_out_queues_map_;
  std::vector<std::thread> rx_service_threads_;
  std::vector<std::thread> tx_service_threads_;
  std::unordered_map<uint32_t, std::shared_ptr<RivermaxManagerRxService>> rx_services_;
  std::unordered_map<uint32_t, std::shared_ptr<RivermaxManagerTxService>> tx_services_;
  BurstParams burst_tx_pool[MAX_TX_BURST];
  std::atomic<uint32_t> burst_tx_idx{0};
  bool initialized_ = false;
};

std::atomic<bool> force_quit = false;

bool RivermaxMgr::RivermaxMgrImpl::set_config_and_initialize(const NetworkConfig& cfg) {
  if (this->initialized_) {
    HOLOSCAN_LOG_INFO("Rivermax ANO Manager already initialized");
    return true;
  }
  cfg_ = &cfg;

  // Start Initialize in a separate thread so it doesn't set the affinity for the
  // whole application
  std::thread t(&RivermaxMgr::RivermaxMgrImpl::initialize, this);
  t.join();

  if (!this->initialized_) {
    HOLOSCAN_LOG_ERROR("Failed to initialize Rivermax Advanced Network Manager");
    return false;
  }

  run();

  return true;
}

void RivermaxMgr::RivermaxMgrImpl::initialize() {
  RivermaxConfigContainer config_manager;

  bool res = config_manager.parse_configuration(*cfg_);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse configuration for Rivermax advanced_network Manager");
    return;
  }
  HOLOSCAN_LOG_INFO("Setting Rivermax Log Level to: {}",
                    holoscan::advanced_network::RivermaxLogLevel::to_description_string(
                        config_manager.get_rivermax_log_level()));
  rivermax_setparam("RIVERMAX_LOG_LEVEL",
                    holoscan::advanced_network::RivermaxLogLevel::to_cmd_string(
                        config_manager.get_rivermax_log_level()),
                    true);

  auto rx_config_manager = std::dynamic_pointer_cast<RxConfigManager>(
      config_manager.get_config_manager(RivermaxConfigContainer::ConfigType::RX));

  if (rx_config_manager) {
    for (const auto& config : *rx_config_manager) {
      res = initialize_rx_service(config.first, config.second);
      if (!res) {
        HOLOSCAN_LOG_ERROR("Failed to initialize RX service for config ID {}", config.first);
        return;
      }
    }
  }

  auto tx_config_manager = std::dynamic_pointer_cast<TxConfigManager>(
      config_manager.get_config_manager(RivermaxConfigContainer::ConfigType::TX));

  if (tx_config_manager) {
    for (const auto& config : *tx_config_manager) {
      res = initialize_tx_service(config.first, config.second);
      if (!res) {
        HOLOSCAN_LOG_ERROR("Failed to initialize TX service for config ID {}", config.first);
        return;
      }
    }
  }

  for (int i = 0; i < MAX_TX_BURST; i++) {
    burst_tx_pool[i].hdr.hdr.port_id = 0;
    burst_tx_pool[i].hdr.hdr.q_id = 0;
    burst_tx_pool[i].hdr.hdr.num_pkts = MAX_NUM_OF_FRAMES_IN_BURST;
    burst_tx_pool[i].pkts[0] = new void*[MAX_NUM_OF_FRAMES_IN_BURST];
    burst_tx_pool[i].pkt_lens[0] = new uint32_t[MAX_NUM_OF_FRAMES_IN_BURST];
    burst_tx_pool[i].pkt_extra_info = new void*[MAX_NUM_OF_FRAMES_IN_BURST];
  }

  this->initialized_ = true;
}

// Extract configuration finding logic
template<typename ConfigType>
ConfigType* RivermaxMgr::RivermaxMgrImpl::find_rivermax_config(uint32_t service_id, QueueConfigType expected_type) {
  // Temporary solution to extract the specific queue configuration from the NetworkConfig
  for (const auto& intf : cfg_->ifs_) {
    for (const auto& rx_queue : intf.rx_.queues_) {
      if (!rx_queue.common_.extra_queue_config_) continue;

      auto* config = dynamic_cast<ConfigType*>(rx_queue.common_.extra_queue_config_);
      if (config && config->get_type() == expected_type && rx_queue.common_.id_ == service_id) {
        return config;
      }
    }
  }
  return nullptr;
}

template<typename ConfigType>
std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>>
RivermaxMgr::RivermaxMgrImpl::create_burst_queues_for_streams(const ConfigType& rivermax_config,
                                                              uint32_t service_id) {
  HOLOSCAN_LOG_INFO("Found {} threads in YAML configuration for service {}",
                    rivermax_config.thread_settings.size(), service_id);

  std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>> queue_map;

  for (const auto& thread : rivermax_config.thread_settings) {
    HOLOSCAN_LOG_INFO("Thread {} has {} streams",
                      thread.thread_id, thread.stream_network_settings.size());

    for (const auto& stream : thread.stream_network_settings) {
      if (queue_map.find(stream.stream_id) != queue_map.end()) {
        HOLOSCAN_LOG_WARN("Duplicate stream ID {} found in configuration for service {}. "
                          "Each stream ID must be unique. Skipping duplicate.",
                          stream.stream_id, service_id);
        continue;
      }
      queue_map[stream.stream_id] = std::make_shared<AnoBurstsQueue>();
    }
  }

  return queue_map;
}

template<typename ConfigType, typename BuilderType, typename ServiceType>
bool RivermaxMgr::RivermaxMgrImpl::common_initialize_rx_service(
    uint32_t service_id, std::shared_ptr<ConfigBuilderHolder> config_holder,
    QueueConfigType expected_type, const std::string& service_name) {
  auto typed_holder = std::dynamic_pointer_cast<TypedConfigBuilderHolder<BuilderType>>(config_holder);
  if (!typed_holder) {
    HOLOSCAN_LOG_ERROR("Failed to cast to TypedConfigBuilderHolder<{}>", typeid(BuilderType).name());
    return false;
  }

  auto builder = typed_holder->get_config_builder();
  if (!builder) {
    HOLOSCAN_LOG_ERROR("Failed to get builder for {}", service_name);
    return false;
  }

  // Extract common configuration logic
  auto rivermax_config = find_rivermax_config<ConfigType>(service_id, expected_type);
  if (!rivermax_config) {
    HOLOSCAN_LOG_ERROR("Could not find {} configuration in YAML for service {}",
                       service_name, service_id);
    return false;
  }

  auto queue_map = create_burst_queues_for_streams(*rivermax_config, service_id);
  rx_bursts_out_queues_map_[service_id] = std::move(queue_map);

  auto rx_service = std::make_shared<ServiceType>(
      service_id, builder, rx_bursts_out_queues_map_[service_id]);

  if (!rx_service->initialize()) {
    HOLOSCAN_LOG_ERROR("Failed to initialize {} service", service_name);
    return false;
  }

  rx_services_[service_id] = std::move(rx_service);
  return true;
}

bool RivermaxMgr::RivermaxMgrImpl::initialize_rx_service(
    uint32_t service_id, std::shared_ptr<ConfigBuilderHolder> config_holder) {
  if (config_holder == nullptr) {
    HOLOSCAN_LOG_ERROR("Config holder is null");
    return false;
  }
  auto config_type = config_holder->get_type();

  switch (config_type) {
    case QueueConfigType::IPOReceiver:
      HOLOSCAN_LOG_INFO("Initializing IPOReceiver:{}", service_id);
      return common_initialize_rx_service<
          RivermaxIPOReceiverQueueConfig,
          RivermaxQueueToIPOReceiverSettingsBuilder,
          IPOReceiverService>(service_id, config_holder, config_type, "IPOReceiver");
    case QueueConfigType::RTPReceiver:
      HOLOSCAN_LOG_INFO("Initializing RTPReceiver:{}", service_id);
      return common_initialize_rx_service<
          RivermaxRTPReceiverQueueConfig,
          RivermaxQueueToRTPReceiverSettingsBuilder,
          RTPReceiverService>(service_id, config_holder, config_type, "RTPReceiver");
    default:
      HOLOSCAN_LOG_ERROR("Unsupported Rx Service configuration type: {}",
                         queue_config_type_to_string(config_type));
      return false;
  }
}

bool RivermaxMgr::RivermaxMgrImpl::initialize_tx_service(
    uint32_t service_id, std::shared_ptr<ConfigBuilderHolder> config_holder) {
  if (config_holder == nullptr) {
    HOLOSCAN_LOG_ERROR("Config holder is null");
    return false;
  }

  std::shared_ptr<RivermaxManagerTxService> tx_service;

  auto config_type = config_holder->get_type();
  if (config_type == QueueConfigType::MediaFrameSender) {
    HOLOSCAN_LOG_INFO("Initializing MediaSender:{}", service_id);

    auto typed_holder = std::dynamic_pointer_cast<
        TypedConfigBuilderHolder<RivermaxQueueToMediaSenderSettingsBuilder>>(config_holder);

    if (!typed_holder) {
      HOLOSCAN_LOG_ERROR(
          "Failed to cast to TypedConfigBuilderHolder<RivermaxQueueToMediaSenderSettingsBuilder>");
      return false;
    }

    auto media_sender_builder = typed_holder->get_config_builder();
    if (!media_sender_builder) {
      HOLOSCAN_LOG_ERROR("Failed to get RivermaxQueueToMediaSenderSettingsBuilder");
      return false;
    }

    auto dummy_sender = media_sender_builder->dummy_sender_;
    if (dummy_sender) {
      HOLOSCAN_LOG_INFO("Initializing Media Mock Sender :{}", service_id);
      tx_service = std::make_shared<MediaSenderMockService>(service_id, media_sender_builder);
    } else {
      if (!media_sender_builder->use_internal_memory_pool_) {
        HOLOSCAN_LOG_INFO("Initializing Media Frame Zero Copy Sender :{}", service_id);
        tx_service = std::make_shared<MediaSenderZeroCopyService>(service_id, media_sender_builder);
      } else {
        HOLOSCAN_LOG_INFO("Initializing Media Frame Sender :{}", service_id);
        tx_service = std::make_shared<MediaSenderService>(service_id, media_sender_builder);
      }
    }
  } else {
    HOLOSCAN_LOG_ERROR("Unsupported Tx Service configuration type: {}",
                       queue_config_type_to_string(config_type));
    return false;
  }

  if (!tx_service->initialize()) {
    HOLOSCAN_LOG_ERROR("Failed to initialize TX service");
    return false;
  }

  tx_services_[service_id] = std::move(tx_service);
  return true;
}

RivermaxMgr::RivermaxMgrImpl::~RivermaxMgrImpl() {}

void RivermaxMgr::RivermaxMgrImpl::run() {
  std::size_t num_services = rx_services_.size();
  if (num_services > 0) { HOLOSCAN_LOG_INFO("Starting {} RX Services", num_services); }

  for (const auto& entry : rx_services_) {
    uint32_t key = entry.first;
    auto& rx_service = entry.second;
    rx_service_threads_.emplace_back(
        [rx_service_ptr = rx_service.get()]() { rx_service_ptr->run(); });
  }

  num_services = tx_services_.size();
  if (num_services > 0) { HOLOSCAN_LOG_INFO("Starting {} TX Services", num_services); }

  for (const auto& entry : tx_services_) {
    uint32_t key = entry.first;
    auto& tx_service = entry.second;
    tx_service_threads_.emplace_back(
        [tx_service_ptr = tx_service.get()]() { tx_service_ptr->run(); });
  }

  HOLOSCAN_LOG_INFO("Done starting workers");
}

void* RivermaxMgr::RivermaxMgrImpl::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
  return burst->pkts[seg][idx];
}

void* RivermaxMgr::RivermaxMgrImpl::get_packet_ptr(BurstParams* burst, int idx) {
  return burst->pkts[0][idx];
}

uint16_t RivermaxMgr::RivermaxMgrImpl::get_segment_packet_length(BurstParams* burst, int seg,
                                                                 int idx) {
  return burst->pkt_lens[seg][idx];
}

uint16_t RivermaxMgr::RivermaxMgrImpl::get_packet_length(BurstParams* burst, int idx) {
  return burst->pkt_lens[0][idx];
}

void* RivermaxMgr::RivermaxMgrImpl::get_packet_extra_info(BurstParams* burst, int idx) {
  RivermaxBurst* rivermax_burst = static_cast<RivermaxBurst*>(burst);
  if (rivermax_burst->is_packet_info_per_packet()) return burst->pkt_extra_info[idx];
  return nullptr;
}

Status RivermaxMgr::RivermaxMgrImpl::set_packet_tx_time(BurstParams* burst, int idx,
                                                        uint64_t timestamp) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::get_tx_packet_burst(BurstParams* burst) {
  uint32_t key =
      RivermaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  auto it = tx_services_.find(key);
  if (it == tx_services_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Invalid Port ID {}, Queue ID {} combination", burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return Status::INVALID_PARAMETER;
  }

  return it->second->get_tx_packet_burst(burst);
}

Status RivermaxMgr::RivermaxMgrImpl::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_ipv4_header(BurstParams* burst, int idx, int ip_len,
                                                     uint8_t proto, unsigned int src_host,
                                                     unsigned int dst_host) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_udp_header(BurstParams* burst, int idx, int udp_len,
                                                    uint16_t src_port, uint16_t dst_port) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_udp_payload(BurstParams* burst, int idx, void* data,
                                                     int len) {
  return Status::SUCCESS;
}

bool RivermaxMgr::RivermaxMgrImpl::is_tx_burst_available(BurstParams* burst) {
  uint32_t key =
      RivermaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  auto it = tx_services_.find(key);
  if (it == tx_services_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Invalid Port ID {}, Queue ID {} combination", burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return false;
  }

  return it->second->is_tx_burst_available(burst);
}

Status RivermaxMgr::RivermaxMgrImpl::set_packet_lengths(BurstParams* burst, int idx,
                                                        const std::initializer_list<int>& lens) {
  return Status::SUCCESS;
}

void RivermaxMgr::RivermaxMgrImpl::free_all_segment_packets(BurstParams* burst, int seg) {}

void RivermaxMgr::RivermaxMgrImpl::free_all_packets(BurstParams* burst) {}

void RivermaxMgr::RivermaxMgrImpl::free_packet_segment(BurstParams* burst, int seg, int pkt) {}

void RivermaxMgr::RivermaxMgrImpl::free_packet(BurstParams* burst, int pkt) {}

void RivermaxMgr::RivermaxMgrImpl::free_rx_burst(BurstParams* burst) {
  uint32_t key =
      RivermaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  auto it = rx_services_.find(key);
  if (it == rx_services_.end()) {
    HOLOSCAN_LOG_ERROR("Rivermax Service is not initialized");
    return;
  }

  it->second->free_rx_burst(burst);
}

void RivermaxMgr::RivermaxMgrImpl::free_tx_burst(BurstParams* burst) {
  uint32_t key =
      RivermaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  auto it = tx_services_.find(key);
  if (it == tx_services_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Invalid Port ID {}, Queue ID {} combination", burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return;
  }

  it->second->free_tx_burst(burst);
}

Status RivermaxMgr::RivermaxMgrImpl::get_rx_burst(BurstParams** burst, int port, int q, int stream_id) {
  uint32_t service_id = RivermaxBurst::burst_tag_from_port_and_queue_id(port, q);
  auto queue_it = rx_bursts_out_queues_map_.find(service_id);

  if (queue_it == rx_bursts_out_queues_map_.end()) {
    HOLOSCAN_LOG_ERROR(
        "No Rx queue found for Rivermax service (port {}, queue {}). "
        "Check config.",
        port,
        q);
    return Status::INVALID_PARAMETER;
  }
  if (queue_it->second.find(stream_id) == queue_it->second.end()) {
    HOLOSCAN_LOG_ERROR("No Rx queue found for stream_id {}", stream_id);
    return Status::INVALID_PARAMETER;
  }

  auto out_burst_shared = queue_it->second[stream_id]->dequeue_burst();
  if (out_burst_shared == nullptr) { return Status::NULL_PTR; }
  *burst = out_burst_shared.get();
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::get_rx_burst(BurstParams** burst, int port, int q) {
  // dequeue burst from any stream
  uint32_t service_id = RivermaxBurst::burst_tag_from_port_and_queue_id(port, q);
  auto queue_it = rx_bursts_out_queues_map_.find(service_id);

  if (queue_it == rx_bursts_out_queues_map_.end()) {
    HOLOSCAN_LOG_ERROR(
        "No Rx queue found for Rivermax service (port {}, queue {}). "
        "Check config.",
        port,
        q);
    return Status::INVALID_PARAMETER;
  }
  if (queue_it->second.empty()) {
    HOLOSCAN_LOG_ERROR("No Rx queues found for service_id {}", service_id);
    return Status::INVALID_PARAMETER;
  }
  // get burst from a random stream
  auto number_of_streams = queue_it->second.size();
  if (number_of_streams == 0) {
    HOLOSCAN_LOG_ERROR("No Rx queues found for service_id {}", service_id);
    return Status::INVALID_PARAMETER;
  }
  auto it = queue_it->second.begin();
  std::advance(it, rand() % number_of_streams);
  auto out_burst_shared = it->second->dequeue_burst();
  if (out_burst_shared == nullptr) { return Status::NULL_PTR; }
  *burst = out_burst_shared.get();
  return Status::SUCCESS;
}

void RivermaxMgr::RivermaxMgrImpl::free_rx_metadata(BurstParams* burst) {}

void RivermaxMgr::RivermaxMgrImpl::free_tx_metadata(BurstParams* burst) {}

Status RivermaxMgr::RivermaxMgrImpl::get_tx_metadata_buffer(BurstParams** burst) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::send_tx_burst(BurstParams* burst) {
  uint32_t key =
      RivermaxBurst::burst_tag_from_port_and_queue_id(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);

  auto it = tx_services_.find(key);
  if (it == tx_services_.end()) {
    HOLOSCAN_LOG_ERROR(
        "Invalid Port ID {}, Queue ID {} combination", burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return Status::INVALID_PARAMETER;
  }

  return it->second->send_tx_burst(burst);
}

void RivermaxMgr::RivermaxMgrImpl::shutdown() {
  if (force_quit.load()) { return; }
  HOLOSCAN_LOG_INFO("Advanced Network Rivermax manager shutting down");
  force_quit.store(true);
  print_stats();
  kill(getpid(), SIGINT);

  // Shut down all services
  for (auto& service_pair : rx_services_) { service_pair.second->shutdown(); }

  for (auto& service_pair : tx_services_) { service_pair.second->shutdown(); }

  for (auto& [service_id, rx_bursts_out_queue_service_map] : rx_bursts_out_queues_map_) {
    for (auto& [stream_id, rx_bursts_out_queue] : rx_bursts_out_queue_service_map) {
      rx_bursts_out_queue->stop();
      rx_bursts_out_queue->clear();
    }
  }

  for (auto& rx_service_thread : rx_service_threads_) {
    if (rx_service_thread.joinable()) { rx_service_thread.join(); }
  }
  for (auto& tx_service_thread : tx_service_threads_) {
    if (tx_service_thread.joinable()) { tx_service_thread.join(); }
  }
  HOLOSCAN_LOG_INFO("All service threads finished");
  rx_bursts_out_queues_map_.clear();
  rx_services_.clear();

  tx_services_.clear();
}

void RivermaxMgr::RivermaxMgrImpl::print_stats() {
  std::stringstream ss;
  ss << std::endl;
  ss << "RIVERMAX Advanced Network Manager Statistics" << std::endl;
  ss << "====================" << std::endl;
  ss << "Service Statistics" << std::endl;
  ss << "----------------" << std::endl;

  for (const auto& entry : rx_services_) { entry.second->print_stats(ss); }

  HOLOSCAN_LOG_INFO(ss.str());
}

uint64_t RivermaxMgr::RivermaxMgrImpl::get_burst_tot_byte(BurstParams* burst) {
  return 0;
}

BurstParams* RivermaxMgr::RivermaxMgrImpl::create_tx_burst_params() {
  auto burst_idx = burst_tx_idx.fetch_add(1);
  return &(burst_tx_pool[burst_idx % MAX_TX_BURST]);
}

Status RivermaxMgr::RivermaxMgrImpl::get_mac_addr(int port, char* mac) {
  return Status::NOT_SUPPORTED;
}

RivermaxMgr::RivermaxMgr() : pImpl(std::make_unique<RivermaxMgrImpl>()) {}

RivermaxMgr::~RivermaxMgr() = default;

bool RivermaxMgr::set_config_and_initialize(const NetworkConfig& cfg) {
  std::lock_guard<std::mutex> lock(initialization_mutex_);

  if (this->initialized_) {
    HOLOSCAN_LOG_INFO("Rivermax Advanced Network Manager has been already initialized");
    return true;
  }

  cfg_ = cfg;

  int port_id = 0;
  for (auto& intf : cfg_.ifs_) {
    intf.port_id_ = port_id++;
    HOLOSCAN_LOG_INFO("{} ({}): assigned port ID {}", intf.name_, intf.address_, intf.port_id_);
  }

  this->initialized_ = pImpl->set_config_and_initialize(cfg_);

  if (this->initialized_) {
    HOLOSCAN_LOG_INFO("Rivermax ANO Manager initialized successfully");
  } else {
    HOLOSCAN_LOG_ERROR("Failed to initialize Rivermax ANO Manager");
  }

  return this->initialized_;
}

void RivermaxMgr::initialize() {
  pImpl->initialize();
}

void RivermaxMgr::run() {
  pImpl->run();
}

Status RivermaxMgr::parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q) {
  return RivermaxConfigParser::parse_rx_queue_rivermax_config(q_item, q);
}

Status RivermaxMgr::parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q) {
  return RivermaxConfigParser::parse_tx_queue_rivermax_config(q_item, q);
}

void* RivermaxMgr::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
  return pImpl->get_segment_packet_ptr(burst, seg, idx);
}

void* RivermaxMgr::get_packet_ptr(BurstParams* burst, int idx) {
  return pImpl->get_packet_ptr(burst, idx);
}

uint16_t RivermaxMgr::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
  return pImpl->get_segment_packet_length(burst, seg, idx);
}

uint16_t RivermaxMgr::get_packet_length(BurstParams* burst, int idx) {
  return pImpl->get_packet_length(burst, idx);
}

uint16_t RivermaxMgr::get_packet_flow_id(BurstParams* burst, int idx) {
  return 0;
}

void* RivermaxMgr::get_packet_extra_info(BurstParams* burst, int idx) {
  return pImpl->get_packet_extra_info(burst, idx);
}

Status RivermaxMgr::get_tx_packet_burst(BurstParams* burst) {
  return pImpl->get_tx_packet_burst(burst);
}

Status RivermaxMgr::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  return pImpl->set_eth_header(burst, idx, dst_addr);
}

Status RivermaxMgr::set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                    unsigned int src_host, unsigned int dst_host) {
  return pImpl->set_ipv4_header(burst, idx, ip_len, proto, src_host, dst_host);
}

Status RivermaxMgr::set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                   uint16_t dst_port) {
  return pImpl->set_udp_header(burst, idx, udp_len, src_port, dst_port);
}

Status RivermaxMgr::set_udp_payload(BurstParams* burst, int idx, void* data, int len) {
  return pImpl->set_udp_payload(burst, idx, data, len);
}

bool RivermaxMgr::is_tx_burst_available(BurstParams* burst) {
  return pImpl->is_tx_burst_available(burst);
}

Status RivermaxMgr::set_packet_lengths(BurstParams* burst, int idx,
                                       const std::initializer_list<int>& lens) {
  return pImpl->set_packet_lengths(burst, idx, lens);
}

void RivermaxMgr::free_all_segment_packets(BurstParams* burst, int seg) {
  pImpl->free_all_segment_packets(burst, seg);
}

void RivermaxMgr::free_all_packets(BurstParams* burst) {
  pImpl->free_all_packets(burst);
}

void RivermaxMgr::free_packet_segment(BurstParams* burst, int seg, int pkt) {
  pImpl->free_packet_segment(burst, seg, pkt);
}

void RivermaxMgr::free_packet(BurstParams* burst, int pkt) {
  pImpl->free_packet(burst, pkt);
}

void RivermaxMgr::free_rx_burst(BurstParams* burst) {
  pImpl->free_rx_burst(burst);
}

void RivermaxMgr::free_tx_burst(BurstParams* burst) {
  pImpl->free_tx_burst(burst);
}

Status RivermaxMgr::get_rx_burst(BurstParams** burst, int port, int q, int stream_id) {
  return pImpl->get_rx_burst(burst, port, q, stream_id);
}

Status RivermaxMgr::get_rx_burst(BurstParams** burst, int port, int q) {
  return pImpl->get_rx_burst(burst, port, q);
}

Status RivermaxMgr::set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp) {
  return pImpl->set_packet_tx_time(burst, idx, timestamp);
}

void RivermaxMgr::free_rx_metadata(BurstParams* burst) {
  pImpl->free_rx_metadata(burst);
}

void RivermaxMgr::free_tx_metadata(BurstParams* burst) {
  pImpl->free_tx_metadata(burst);
}

Status RivermaxMgr::get_tx_metadata_buffer(BurstParams** burst) {
  return pImpl->get_tx_metadata_buffer(burst);
}

Status RivermaxMgr::send_tx_burst(BurstParams* burst) {
  return pImpl->send_tx_burst(burst);
}

void RivermaxMgr::shutdown() {
  pImpl->shutdown();
}

void RivermaxMgr::print_stats() {
  pImpl->print_stats();
}

uint64_t RivermaxMgr::get_burst_tot_byte(BurstParams* burst) {
  return pImpl->get_burst_tot_byte(burst);
}

BurstParams* RivermaxMgr::create_tx_burst_params() {
  return pImpl->create_tx_burst_params();
}

Status RivermaxMgr::get_mac_addr(int port, char* mac) {
  return pImpl->get_mac_addr(port, mac);
}

};  // namespace holoscan::advanced_network
