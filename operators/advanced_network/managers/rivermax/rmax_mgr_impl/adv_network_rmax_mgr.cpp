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

#include "adv_network_rmax_mgr.h"
#include "rmax_mgr_impl/rmax_config_manager.h"
#include "rmax_mgr_impl/burst_manager.h"
#include "rmax_mgr_impl/packet_processor.h"
#include "rmax_mgr_impl/rmax_chunk_consumer_ano.h"
#include "rmax_mgr_impl/stats_printer.h"
#include "rmax_ipo_receiver_service.h"
#include <holoscan/logger/logger.hpp>
#include "rt_threads.h"

using namespace std::chrono;

namespace holoscan::ops {

using namespace ral::services::rmax_ipo_receiver;

/**
 * A map of log level to a tuple of the description and command strings.
 */
const std::unordered_map<RmaxLogLevel::Level, std::tuple<std::string, std::string>>
    RmaxLogLevel::level_to_cmd_map = {
        {TRACE, {"Trace", "0"}},
        {DEBUG, {"Debug", "1"}},
        {INFO, {"Info", "2"}},
        {WARN, {"Warning", "3"}},
        {ERROR, {"Error", "4"}},
        {CRITICAL, {"Critical", "5"}},
        {OFF, {"Disabled", "6"}},
};

/**
 * A map of Ano log level to Rmax log level.
 */
const std::unordered_map<AnoLogLevel::Level, RmaxLogLevel::Level>
    RmaxLogLevel::ano_to_rmax_log_level_map = {
        {AnoLogLevel::TRACE, TRACE},
        {AnoLogLevel::DEBUG, DEBUG},
        {AnoLogLevel::INFO, INFO},
        {AnoLogLevel::WARN, WARN},
        {AnoLogLevel::ERROR, ERROR},
        {AnoLogLevel::CRITICAL, CRITICAL},
        {AnoLogLevel::OFF, OFF},
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
  static void flush_packets(int port);
  void setup_accurate_send_scheduling_mask();
  int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
  void initialize_rx_service(uint32_t service_id, const ExtRmaxIPOReceiverConfig& config);

 private:
  static constexpr int DEFAULT_NUM_RX_BURST = 64;

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
};

std::atomic<bool> force_quit = false;

/**
 * @brief Sets the configuration and initializes the Rmax manager.
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

    if (!this->initialized_) {
      HOLOSCAN_LOG_ERROR("Failed to initialize Rivermax ANO Manager");
      return false;
    }
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
  rx_bursts_out_queue = std::make_shared<AnoBurstsQueue>();

  rmax_apps_lib = std::make_shared<ral::lib::RmaxAppsLibFacade>();
  RmaxConfigContainer config_manager(rmax_apps_lib);

  bool res = config_manager.parse_configuration(cfg_);
  if (!res) {
    HOLOSCAN_LOG_ERROR("Failed to parse configuration for Rivermax ANO Manager");
    return;
  }
  HOLOSCAN_LOG_INFO(
      "Setting Rivermax Log Level to: {}",
      holoscan::ops::RmaxLogLevel::to_description_string(config_manager.get_rmax_log_level()));
  rivermax_setparam("RIVERMAX_LOG_LEVEL",
                    holoscan::ops::RmaxLogLevel::to_cmd_string(config_manager.get_rmax_log_level()),
                    true);

  auto rx_config_manager = std::dynamic_pointer_cast<RxConfigManager>(
      config_manager.get_config_manager(RmaxConfigContainer::ConfigType::RX));

  if (rx_config_manager) {
    for (const auto& config : *rx_config_manager) {
      initialize_rx_service(config.first, config.second);
    }
  }

  this->initialized_ = true;
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

  auto rx_service = std::make_unique<RmaxIPOReceiverService>(config);
  auto init_status = rx_service->get_init_status();
  if (init_status != ReturnStatus::obj_init_success) {
    HOLOSCAN_LOG_ERROR("Failed to initialize RX service, status: {}", (int)init_status);
    return;
  }

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

  rx_services[service_id]->set_chunk_consumer(rmax_chunk_consumers[service_id].get());
}

/**
 * @brief Destructor for RmaxMgrImpl.
 *
 * This destructor prints the statistics and joins all RX service threads
 * before destroying the Rmax manager implementation.
 */
RmaxMgr::RmaxMgrImpl::~RmaxMgrImpl() {
  for (auto& rx_service_thread : rx_service_threads) {
    if (rx_service_thread.joinable()) { rx_service_thread.join(); }
  }
  for (auto& [service_id, rx_service] : rx_services) {
    if (rx_service) { rx_service->set_chunk_consumer(nullptr); }
  }
  rx_services.clear();

  rx_burst_managers.clear();

  rx_packet_processors.clear();

  rmax_chunk_consumers.clear();

  rx_bursts_out_queue->clear();

  rx_bursts_out_queue.reset();
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

/**
 * @brief Runs the Rmax manager.
 *
 * This function starts the RX services by iterating over the RX services map
 * and launching each service in a separate thread. It ensures that each service
 * is properly initialized before running.
 */
void RmaxMgr::RmaxMgrImpl::run() {
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
  auto it = std::find_if(cfg_.ifs_.begin(), cfg_.ifs_.end(), [&name](const auto& intf) {
    return name == intf.address_;
  });

  if (it != cfg_.ifs_.end()) { return it->port_id_; }

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
    HOLOSCAN_LOG_INFO("ANO Rivermax manager shutting down");
    force_quit.store(false);
    std::raise(SIGINT);
  }
}

/**
 * @brief Prints the statistics of the Rmax manager.
 */
void RmaxMgr::RmaxMgrImpl::print_stats() {
  std::stringstream ss;
  IpoRxStatsPrinter::print_total_stats(ss, rx_services);
  HOLOSCAN_LOG_INFO(ss.str());
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
  auto it = std::find_if(cfg_.ifs_.begin(), cfg_.ifs_.end(), [&addr](const auto& intf) {
    return intf.address_ == addr;
  });

  if (it != cfg_.ifs_.end()) { return it->port_id_; }
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
  return RmaxConfigParser::parse_rx_queue_rivermax_config(q_item, q);
}

/**
 * @brief Parses the TX queue Rivermax configuration.
 *
 * @param q_item The YAML node containing the queue item.
 * @param q The TX queue configuration to be populated.
 * @return AdvNetStatus indicating the success or failure of the operation.
 */
AdvNetStatus RmaxMgr::parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q) {
  return RmaxConfigParser::parse_tx_queue_rivermax_config(q_item, q);
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
 * @brief Gets the flow ID of a packet. Currently returns 0 for the Rivermax backend
 *
 * @param burst The burst parameters.
 * @param idx The packet index.
 * @return Flow ID of the packet
 */
uint16_t RmaxMgr::get_pkt_flow_id(AdvNetBurstParams* burst, int idx) {
  return 0;
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
