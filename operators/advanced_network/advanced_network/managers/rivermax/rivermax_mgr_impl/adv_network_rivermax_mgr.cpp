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
#include "rdk/apps/rmax_ipo_receiver/rmax_ipo_receiver.h"
#include "rdk/apps/rmax_rtp_receiver/rmax_rtp_receiver.h"
#include "rdk/apps/rmax_xstream_media_sender/rmax_xstream_media_sender.h"

#include "adv_network_rivermax_mgr.h"
#include "rivermax_mgr_impl/rivermax_config_manager.h"
#include "rivermax_mgr_impl/burst_manager.h"
#include "rivermax_mgr_impl/packet_processor.h"
#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"
#include "rivermax_mgr_impl/stats_printer.h"

using namespace std::chrono;

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::apps;
using namespace rivermax::dev_kit::apps::rmax_ipo_receiver;
using namespace rivermax::dev_kit::apps::rmax_rtp_receiver;
using namespace rivermax::dev_kit::apps::rmax_xstream_media_sender;

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
  static void flush_packets(int port);
  void setup_accurate_send_scheduling_mask();
  int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
  bool initialize_rx_service(uint32_t service_id,
                             std::shared_ptr<ConfigBuilderHolder> config_holder);
  bool initialize_tx_service(uint32_t service_id,
                              std::shared_ptr<ConfigBuilderHolder> config_holder);

 private:
  static constexpr int DEFAULT_NUM_RX_BURST = 64;

  NetworkConfig cfg_;
  std::unordered_map<uint32_t, std::unique_ptr<RmaxReceiverBaseApp>> rx_services_;
  std::unordered_map<uint32_t, std::shared_ptr<RxBurstsManager>> rx_burst_managers_;
  std::unordered_map<uint32_t, std::shared_ptr<RxPacketProcessor>> rx_packet_processors_;
  std::unordered_map<uint32_t, std::shared_ptr<AnoBurstsQueue>> rx_bursts_out_queues_map_;
  std::vector<std::thread> rx_service_threads_;
  std::unordered_map<uint32_t, std::unique_ptr<MediaSenderApp>> tx_services_;
  std::vector<std::thread> tx_service_threads_;
    bool initialized_ = false;
};

std::atomic<bool> force_quit = false;

bool RivermaxMgr::RivermaxMgrImpl::set_config_and_initialize(const NetworkConfig& cfg) {
  if (!this->initialized_) {
    cfg_ = cfg;

    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&RivermaxMgr::RivermaxMgrImpl::initialize, this);
    t.join();

    if (!this->initialized_) {
      HOLOSCAN_LOG_ERROR("Failed to initialize Rivermax advanced_network Manager");
      return false;
    }
    run();
  }

  return true;
}

void RivermaxMgr::RivermaxMgrImpl::initialize() {
  int port_id = 0;
  for (auto& intf : cfg_.ifs_) {
    intf.port_id_ = port_id++;
    HOLOSCAN_LOG_INFO("{} ({}): assigned port ID {}", intf.name_, intf.address_, intf.port_id_);
  }

  RivermaxConfigContainer config_manager;

  bool res = config_manager.parse_configuration(cfg_);
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

  this->initialized_ = true;
}

bool RivermaxMgr::RivermaxMgrImpl::initialize_rx_service(
    uint32_t service_id, std::shared_ptr<ConfigBuilderHolder> config_holder) {
  uint16_t port_id = RivermaxBurst::burst_port_id_from_burst_tag(service_id);
  uint16_t queue_id = RivermaxBurst::burst_queue_id_from_burst_tag(service_id);

  if (config_holder == nullptr) {
    HOLOSCAN_LOG_ERROR("Config holder is null");
    return false;
  }
  std::unique_ptr<RmaxReceiverBaseApp> rx_service;
  bool send_packet_ext_info = false;
  int gpu_id = INVALID_GPU_ID;
  size_t max_chunk_size = 0;
  auto config_type = config_holder->get_type();
  if (config_type == QueueConfigType::IPOReceiver) {
    HOLOSCAN_LOG_INFO("Initializing IPOReceiver:{}", service_id);
    auto typed_holder = std::dynamic_pointer_cast<
        TypedConfigBuilderHolder<RivermaxQueueToIPOReceiverSettingsBuilder>>(config_holder);
    std::dynamic_pointer_cast<RivermaxQueueToIPOReceiverSettingsBuilder>(config_holder);
    if (!typed_holder) {
      HOLOSCAN_LOG_ERROR(
          "Failed to cast to TypedConfigBuilderHolder<RivermaxQueueToIPOReceiverSettingsBuilder>");
      return false;
    }
    auto ipo_receiver_builder = typed_holder->get_config_builder();
    if (!ipo_receiver_builder) {
      HOLOSCAN_LOG_ERROR("Failed to get RivermaxQueueToIPOReceiverSettingsBuilder");
      return false;
    }
    rx_service = std::make_unique<IPOReceiverApp>(ipo_receiver_builder);
    send_packet_ext_info = ipo_receiver_builder->send_packet_ext_info_;
    auto init_status = rx_service->initialize();
    if (init_status != ReturnStatus::obj_init_success) {
      HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
      return false;
    }
    HOLOSCAN_LOG_INFO("IPOReceiver:{} was successfully initialized", service_id);
    gpu_id = ipo_receiver_builder->built_settings_.gpu_id;
    max_chunk_size = ipo_receiver_builder->built_settings_.max_chunk_size;
  } else if (config_type == QueueConfigType::RTPReceiver) {
    HOLOSCAN_LOG_INFO("Initializing RTPReceiver:{}", service_id);
    auto typed_holder = std::dynamic_pointer_cast<
        TypedConfigBuilderHolder<RivermaxQueueToRTPReceiverSettingsBuilder>>(config_holder);
    std::dynamic_pointer_cast<RivermaxQueueToRTPReceiverSettingsBuilder>(config_holder);
    if (!typed_holder) {
      HOLOSCAN_LOG_ERROR(
          "Failed to cast to TypedConfigBuilderHolder<RivermaxQueueToRTPReceiverSettingsBuilder>");
      return false;
    }
    auto rtp_receiver_builder = typed_holder->get_config_builder();
    if (!rtp_receiver_builder) {
      HOLOSCAN_LOG_ERROR("Failed to get RivermaxQueueToRTPReceiverSettingsBuilder");
      return false;
    }
    rx_service = std::make_unique<RTPReceiverApp>(rtp_receiver_builder);
    send_packet_ext_info = rtp_receiver_builder->send_packet_ext_info_;
    auto init_status = rx_service->initialize();
    if (init_status != ReturnStatus::obj_init_success) {
      HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
      return false;
    }
    HOLOSCAN_LOG_INFO("RTPReceiver:{} was successfully initialized", service_id);
    gpu_id = rtp_receiver_builder->built_settings_.gpu_id;
    max_chunk_size = rtp_receiver_builder->max_chunk_size_;
  } else {
    HOLOSCAN_LOG_ERROR("Unsupported Rx Service configuration type: {}",
                       queue_config_type_to_string(config_type));
    return false;
  }

  // Create a dedicated queue for this service_id
  auto rx_service_out_queue = std::make_shared<AnoBurstsQueue>();
  rx_bursts_out_queues_map_[service_id] = rx_service_out_queue;

  rx_services_[service_id] = std::move(rx_service);
  rx_burst_managers_[service_id] = std::make_shared<RxBurstsManager>(
      send_packet_ext_info, port_id, queue_id, max_chunk_size, gpu_id, rx_service_out_queue);

  rx_packet_processors_[service_id] =
      std::make_shared<RxPacketProcessor>(rx_burst_managers_[service_id]);

  auto rivermax_chunk_consumer =
      std::make_unique<RivermaxChunkConsumerAno>(rx_packet_processors_[service_id]);

  auto status = rx_services_[service_id]->set_receive_data_consumer(
    0, std::move(rivermax_chunk_consumer));
  if (status != ReturnStatus::success) {
    HOLOSCAN_LOG_ERROR("Failed to set receive data consumer, status: {}", (int)status);
    return false;
  }
  return true;
}

bool RivermaxMgr::RivermaxMgrImpl::initialize_tx_service(
    uint32_t service_id, std::shared_ptr<ConfigBuilderHolder> config_holder) {
  uint16_t port_id = RivermaxBurst::burst_port_id_from_burst_tag(service_id);
  uint16_t queue_id = RivermaxBurst::burst_queue_id_from_burst_tag(service_id);

  if (config_holder == nullptr) {
    HOLOSCAN_LOG_ERROR("Config holder is null");
    return false;
  }
  std::unique_ptr<MediaSenderApp> tx_service;
  bool send_packet_ext_info = false;
  int gpu_id = INVALID_GPU_ID;
  size_t max_chunk_size = 0;
  auto config_type = config_holder->get_type();
  if (config_type == QueueConfigType::MediaFrameSender) {
    HOLOSCAN_LOG_INFO("Initializing MediaSender:{}", service_id);
    auto typed_holder = std::dynamic_pointer_cast<
        TypedConfigBuilderHolder<RivermaxQueueToMediaSenderSettingsBuilder>>(config_holder);
    std::dynamic_pointer_cast<RivermaxQueueToMediaSenderSettingsBuilder>(config_holder);
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
    tx_service = std::make_unique<MediaSenderApp>(media_sender_builder);
    auto init_status = tx_service->initialize();
    if (init_status != ReturnStatus::obj_init_success) {
      HOLOSCAN_LOG_ERROR("Failed to initialize TX service, status: {}", (int)init_status);
      return false;
    }
    HOLOSCAN_LOG_INFO("MediaSender:{} was successfully initialized", service_id);
  } else {
    HOLOSCAN_LOG_ERROR("Unsupported Tx Service configuration type: {}",
                       queue_config_type_to_string(config_type));
    return false;
  }

  tx_services_[service_id] = std::move(tx_service);
  return true;
 }

RivermaxMgr::RivermaxMgrImpl::~RivermaxMgrImpl() {
  for (auto& rx_service_thread : rx_service_threads_) {
    if (rx_service_thread.joinable()) { rx_service_thread.join(); }
  }
  for (auto& tx_service_thread : tx_service_threads_) {
    if (tx_service_thread.joinable()) { tx_service_thread.join(); }
  }

  rx_services_.clear();
  rx_burst_managers_.clear();
  rx_packet_processors_.clear();
  for (auto& [service_id, rx_bursts_out_queue] : rx_bursts_out_queues_map_) {
    rx_bursts_out_queue->clear();
  }
  rx_bursts_out_queues_map_.clear();

  tx_services_.clear();
}

void RivermaxMgr::RivermaxMgrImpl::run() {

  std::size_t num_services = rx_services_.size();
  if (num_services > 0) {
    HOLOSCAN_LOG_INFO("Starting {} RX Services", num_services);
  }

  for (const auto& entry : rx_services_) {
    uint32_t key = entry.first;
    auto& rx_service = entry.second;
    rx_service_threads_.emplace_back([rx_service_ptr = rx_service.get()]() {
      ReturnStatus status = rx_service_ptr->run();
      // TODO: Handle the status if needed
    });
  }

  num_services = tx_services_.size();
  if (num_services > 0) {
    HOLOSCAN_LOG_INFO("Starting {} TX Services", num_services);
  }

  for (const auto& entry : tx_services_) {
    uint32_t key = entry.first;
    auto& tx_service = entry.second;
    tx_service_threads_.emplace_back([tx_service_ptr = tx_service.get()]() {
      ReturnStatus status = tx_service_ptr->run();
      // TODO: Handle the status if needed
    });
  }

  HOLOSCAN_LOG_INFO("Done starting workers");
}

void RivermaxMgr::RivermaxMgrImpl::flush_packets(int port) {
  HOLOSCAN_LOG_INFO("Flushing packet on port {}", port);
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
  return Status::NO_FREE_BURST_BUFFERS;
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
  return false;
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

  if (rx_services_.find(key) == rx_services_.end() ||
      rx_burst_managers_.find(key) == rx_burst_managers_.end()) {
    HOLOSCAN_LOG_ERROR("Rivermax Service is not initialized");
    return;
  }

  auto& rx_service = rx_services_[key];
  auto& rivermax_bursts_manager = rx_burst_managers_[key];

  rivermax_bursts_manager->rx_burst_done(static_cast<RivermaxBurst*>(burst));
}

void RivermaxMgr::RivermaxMgrImpl::free_tx_burst(BurstParams* burst) {}

Status RivermaxMgr::RivermaxMgrImpl::get_rx_burst(BurstParams** burst, int port, int q) {
  uint32_t service_id = RivermaxBurst::burst_tag_from_port_and_queue_id(port, q);
  auto queue_it = rx_bursts_out_queues_map_.find(service_id);

  if (queue_it == rx_bursts_out_queues_map_.end()) {
    HOLOSCAN_LOG_ERROR("No Rx queue found for Rivermax service (port {}, queue {}). "
                       "Check config.", port, q);
    return Status::INVALID_PARAMETER;
  }

  auto out_burst_shared = queue_it->second->dequeue_burst();
  if (out_burst_shared == nullptr) {
    return Status::NULL_PTR;
  }
  *burst = out_burst_shared.get();
  return Status::SUCCESS;
}

void RivermaxMgr::RivermaxMgrImpl::free_rx_metadata(BurstParams* burst) {}

void RivermaxMgr::RivermaxMgrImpl::free_tx_metadata(BurstParams* burst) {}

Status RivermaxMgr::RivermaxMgrImpl::get_tx_metadata_buffer(BurstParams** burst) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::send_tx_burst(BurstParams* burst) {
  return Status::SUCCESS;
}

void RivermaxMgr::RivermaxMgrImpl::shutdown() {
  if (force_quit.load()) { return; }
  HOLOSCAN_LOG_INFO("Advanced Network Rivermax manager shutting down");
  force_quit.store(true);
  print_stats();
  kill(getpid(), SIGINT);
  for (auto& [service_id, rx_bursts_out_queue] : rx_bursts_out_queues_map_) {
    rx_bursts_out_queue->stop();
  }
}

void RivermaxMgr::RivermaxMgrImpl::print_stats() {
  std::stringstream ss;
  RxStatsPrinter::print_total_stats(ss, rx_services_);
  HOLOSCAN_LOG_INFO(ss.str());
}

uint64_t RivermaxMgr::RivermaxMgrImpl::get_burst_tot_byte(BurstParams* burst) {
  return 0;
}

BurstParams* RivermaxMgr::RivermaxMgrImpl::create_tx_burst_params() {
  return new BurstParams();
}

Status RivermaxMgr::RivermaxMgrImpl::get_mac_addr(int port, char* mac) {
  return Status::NOT_SUPPORTED;
}

RivermaxMgr::RivermaxMgr() : pImpl(std::make_unique<RivermaxMgrImpl>()) {}

RivermaxMgr::~RivermaxMgr() = default;

bool RivermaxMgr::set_config_and_initialize(const NetworkConfig& cfg) {
  bool res = true;
  if (!this->initialized_) {
    res = pImpl->set_config_and_initialize(cfg);
    this->initialized_ = res;
  }
  if (this->initialized_) {
    HOLOSCAN_LOG_INFO("Rivermax ANO Manager initialized successfully");
    cfg_ = cfg;
  } else {
    HOLOSCAN_LOG_ERROR("Failed to initialize Rivermax ANO Manager");
  }
  return res;
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
