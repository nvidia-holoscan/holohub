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

#include "adv_network_rivermax_mgr.h"
#include "rivermax_mgr_impl/rivermax_config_manager.h"
#include "rivermax_mgr_impl/burst_manager.h"
#include "rivermax_mgr_impl/packet_processor.h"
#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"
#include "rivermax_mgr_impl/stats_printer.h"
#include "rmax_ipo_receiver_service.h"
#include <holoscan/logger/logger.hpp>
#include "rt_threads.h"

using namespace std::chrono;

namespace holoscan::advanced_network {

using namespace ral::services::rmax_ipo_receiver;

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
  void initialize_rx_service(uint32_t service_id, const ExtRmaxIPOReceiverConfig& config);

 private:
  static constexpr int DEFAULT_NUM_RX_BURST = 64;

  NetworkConfig cfg_;
  std::unordered_map<uint32_t,
                     std::unique_ptr<ral::services::rmax_ipo_receiver::RmaxIPOReceiverService>>
      rx_services;
  std::unordered_map<uint32_t, std::unique_ptr<RivermaxChunkConsumerAno>> rivermax_chunk_consumers;
  std::unordered_map<uint32_t, std::shared_ptr<RxBurstsManager>> rx_burst_managers;
  std::unordered_map<uint32_t, std::shared_ptr<RxPacketProcessor>> rx_packet_processors;
  std::unordered_map<uint32_t, std::shared_ptr<AnoBurstsQueue>> rx_bursts_out_queues_map_;
  std::vector<std::thread> rx_service_threads;
  bool initialized_ = false;
  std::shared_ptr<ral::lib::RmaxAppsLibFacade> rmax_apps_lib = nullptr;
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

  rmax_apps_lib = std::make_shared<ral::lib::RmaxAppsLibFacade>();
  RivermaxConfigContainer config_manager(rmax_apps_lib);

  bool res = config_manager.parse_configuration(cfg_);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse configuration for Rivermax advanced_network Manager");
    return;
  }
  HOLOSCAN_LOG_INFO("Setting Rivermax Log Level to: {}",
                    holoscan::advanced_network::RivermaxLogLevel::to_description_string(
                        config_manager.get_rivermax_log_level()));
  rivermax_setparam(
      "RIVERMAX_LOG_LEVEL",
      holoscan::advanced_network::RivermaxLogLevel::to_cmd_string(
        config_manager.get_rivermax_log_level()), true);

  auto rx_config_manager = std::dynamic_pointer_cast<RxConfigManager>(
      config_manager.get_config_manager(RivermaxConfigContainer::ConfigType::RX));

  if (rx_config_manager) {
    for (const auto& config : *rx_config_manager) {
      initialize_rx_service(config.first, config.second);
    }
  }

  this->initialized_ = true;
}

void RivermaxMgr::RivermaxMgrImpl::initialize_rx_service(uint32_t service_id,
                                                 const ExtRmaxIPOReceiverConfig& config) {
  uint16_t port_id = RivermaxBurst::burst_port_id_from_burst_tag(service_id);
  uint16_t queue_id = RivermaxBurst::burst_queue_id_from_burst_tag(service_id);

  auto rx_service = std::make_unique<RmaxIPOReceiverService>(config);
  auto init_status = rx_service->get_init_status();
  if (init_status != ReturnStatus::obj_init_success) {
    HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
    return;
  }

  // Create a dedicated queue for this service_id
  auto queue = std::make_shared<AnoBurstsQueue>();
  rx_bursts_out_queues_map_[service_id] = queue;

  rx_services[service_id] = std::move(rx_service);
  rx_burst_managers[service_id] = std::make_shared<RxBurstsManager>(config.send_packet_ext_info,
                                                                    port_id,
                                                                    queue_id,
                                                                    config.max_chunk_size,
                                                                    config.app_settings->gpu_id,
                                                                    queue);

  rx_packet_processors[service_id] =
      std::make_shared<RxPacketProcessor>(rx_burst_managers[service_id]);

  rivermax_chunk_consumers[service_id] =
      std::make_unique<RivermaxChunkConsumerAno>(rx_packet_processors[service_id]);

  rx_services[service_id]->set_chunk_consumer(rivermax_chunk_consumers[service_id].get());
}

RivermaxMgr::RivermaxMgrImpl::~RivermaxMgrImpl() {
  for (auto& rx_service_thread : rx_service_threads) {
    if (rx_service_thread.joinable()) { rx_service_thread.join(); }
  }
  for (auto& [service_id, rx_service] : rx_services) {
    if (rx_service) { rx_service->set_chunk_consumer(nullptr); }
  }
  rx_services.clear();

  rx_burst_managers.clear();

  rx_packet_processors.clear();

  rivermax_chunk_consumers.clear();

  for (auto& [service_id, rx_bursts_out_queue] : rx_bursts_out_queues_map_) {
    rx_bursts_out_queue->clear();
  }
  rx_bursts_out_queues_map_.clear();
}

class RmaxServicesSynchronizer : public IRmaxServicesSynchronizer {
 public:
  explicit RmaxServicesSynchronizer(std::size_t count)
      : threshold(count), count(count), generation(0), ready(false), waiting_threads(0) {}

  void wait_for_start() override {
    wait_at_barrier();
    wait_for_signal_to_run();
  }

  void start_all() {
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
    cv.notify_all();
  }

  void wait_until_all_ready_to_run() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return waiting_threads == threshold; });
  }

  void wait_until_all_are_running() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return waiting_threads == 0; });
  }

 private:
  void wait_at_barrier() {
    std::unique_lock<std::mutex> lock(mtx);
    auto gen = generation;
    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
    } else {
      cv.wait(lock, [this, gen] { return gen != generation; });
    }
  }

  void wait_for_signal_to_run() {
    std::unique_lock<std::mutex> lock(mtx);
    waiting_threads++;
    cv.notify_all();
    cv.wait(lock, [this] { return ready; });
    waiting_threads--;
    if (waiting_threads == 0) { cv.notify_all(); }
  }

 private:
  std::mutex mtx;
  std::condition_variable cv;
  std::size_t threshold;
  std::size_t count;
  std::size_t generation;
  bool ready;
  std::size_t waiting_threads;
};

void RivermaxMgr::RivermaxMgrImpl::run() {
  HOLOSCAN_LOG_INFO("Starting RX Services");

  std::size_t num_services = rx_services.size();
  RmaxServicesSynchronizer services_sync(num_services);

  for (const auto& entry : rx_services) {
    uint32_t key = entry.first;
    auto& rx_service = entry.second;
    if (rx_service->get_init_status() != ReturnStatus::obj_init_success) {
      HOLOSCAN_LOG_ERROR("Rx Service failed to initialize, cannot run");
      return;
    }
    rx_service_threads.emplace_back([rx_service_ptr = rx_service.get(), &services_sync]() {
      ReturnStatus status = rx_service_ptr->run(&services_sync);
      // TODO: Handle the status if needed
    });
  }
  services_sync.wait_until_all_ready_to_run();
  services_sync.start_all();
  services_sync.wait_until_all_are_running();

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

uint16_t RivermaxMgr::RivermaxMgrImpl::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
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

Status RivermaxMgr::RivermaxMgrImpl::set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::get_tx_packet_burst(BurstParams* burst) {
  return Status::NO_FREE_BURST_BUFFERS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                             unsigned int src_host, unsigned int dst_host) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_udp_header(BurstParams* burst, int idx, int udp_len,
                                            uint16_t src_port, uint16_t dst_port) {
  return Status::SUCCESS;
}

Status RivermaxMgr::RivermaxMgrImpl::set_udp_payload(BurstParams* burst, int idx, void* data, int len) {
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

  if (rx_services.find(key) == rx_services.end() ||
      rx_burst_managers.find(key) == rx_burst_managers.end()) {
    HOLOSCAN_LOG_ERROR("Rivermax Service is not initialized");
    return;
  }

  auto& rx_service = rx_services[key];
  auto& rivermax_bursts_manager = rx_burst_managers[key];

  if (!(rx_service->is_alive())) {
    HOLOSCAN_LOG_ERROR("Rivermax Service is not initialized");
    return;
  }

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
  if (!force_quit.load()) {
    print_stats();

    HOLOSCAN_LOG_INFO("advanced_network Rivermax manager shutting down");
    force_quit.store(false);
    std::raise(SIGINT);
  }
}

void RivermaxMgr::RivermaxMgrImpl::print_stats() {
  std::stringstream ss;
  IpoRxStatsPrinter::print_total_stats(ss, rx_services);
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
