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

#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_metrics.h>
#include <rte_bitrate.h>
#include <rte_latencystats.h>
#include <rte_flow.h>
#include <rte_gpudev.h>
#include <atomic>
#include <unordered_map>
#include "advanced_network/manager.h"
#include "advanced_network/common.h"
#include "adv_network_dpdk_stats.h"

namespace holoscan::advanced_network {

struct DPDKQueueConfig {
  std::vector<struct rte_mempool*> pools;
  struct rte_eth_rxconf rxconf_qsplit;
  std::vector<union rte_eth_rxseg> rx_useg;
};


class DpdkMgr : public Manager {
 public:
  static_assert(MAX_INTERFACES <= RTE_MAX_ETHPORTS, "Too many interfaces configured");

  DpdkMgr() = default;
  ~DpdkMgr();
  bool set_config_and_initialize(const NetworkConfig& cfg) override;
  void initialize() override;
  void run() override;
  static constexpr int JUMBOFRAME_SIZE = 9100;
  static constexpr int DEFAULT_NUM_TX_BURST = 256;
  static constexpr uint16_t DEFAULT_NUM_RX_BURST = 64;
  uint16_t default_num_rx_desc = 8192;
  uint16_t default_num_tx_desc = 8192;
  int num_ports = 0;
  static constexpr int MEMPOOL_CACHE_SIZE = 32;

  static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
  static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);
  static constexpr uint32_t CPU_PAGE_SIZE = 4096;
  static constexpr int BUFFER_SPLIT_SEGS = 2;
  static constexpr int MAX_ETH_HDR_SIZE = 18;

  void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) override;
  void* get_packet_ptr(BurstParams* burst, int idx) override;
  uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx) override;
  uint32_t get_packet_length(BurstParams* burst, int idx) override;
  uint16_t get_packet_flow_id(BurstParams* burst, int idx) override;
  void* get_packet_extra_info(BurstParams* burst, int idx) override;
  Status get_tx_packet_burst(BurstParams* burst) override;
  Status set_eth_header(BurstParams* burst, int idx, char* dst_addr) override;
  Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                            unsigned int src_host, unsigned int dst_host) override;
  Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                           uint16_t dst_port) override;
  Status set_udp_payload(BurstParams* burst, int idx, void* data, int len) override;
  bool is_tx_burst_available(BurstParams* burst) override;

  Status set_packet_lengths(BurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  void free_all_segment_packets(BurstParams* burst, int seg) override;
  void free_packet_segment(BurstParams* burst, int seg, int pkt) override;
  void free_packet(BurstParams* burst, int pkt) override;
  void free_all_packets(BurstParams* burst) override;
  void free_rx_burst(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;

  Status get_rx_burst(BurstParams** burst, int port, int q) override;
  using holoscan::advanced_network::Manager::get_rx_burst;  // for overloads
  Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_metadata(BurstParams* burst) override;
  void free_tx_metadata(BurstParams* burst) override;
  Status get_tx_metadata_buffer(BurstParams** burst) override;
  Status send_tx_burst(BurstParams* burst) override;
  Status get_mac_addr(int port, char* mac) override;
  void shutdown() override;
  void print_stats() override;
  void adjust_memory_regions() override;
  uint64_t get_burst_tot_byte(BurstParams* burst) override;
  BurstParams* create_tx_burst_params() override;
  bool validate_config() const override;
  uint16_t get_num_rx_queues(int port_id) const override;

 private:
  static void PrintDpdkStats(int port);
  static std::string generate_random_string(int len);
  static int rx_core_worker(void* arg);
  static int rx_core_multi_q_worker(void* arg);
  static int tx_core_worker(void* arg);
  static void flush_packets(int port);
  void setup_accurate_send_scheduling_mask();
  int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
  struct rte_flow* add_flow(int port, const FlowConfig& cfg);
  void create_dummy_rx_q();
  struct rte_flow* add_modify_flow_set(int port, int queue, const char* buf, int len,
                                       Direction direction);

  void apply_tx_offloads(int port);

  std::array<struct rte_ether_addr, MAX_IFS> mac_addrs;
  std::unordered_map<uint32_t, struct rte_ring*> rx_rings;
  struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
  std::unordered_map<uint32_t, struct rte_ring*> tx_rings;
  std::unordered_map<uint32_t, struct rte_mempool*> tx_burst_buffers;
  std::unordered_map<uint32_t, DPDKQueueConfig*> rx_dpdk_q_map_;
  std::unordered_map<uint32_t, DPDKQueueConfig*> tx_dpdk_q_map_;
  std::unordered_map<uint32_t, const RxQueueConfig*> rx_cfg_q_map_;
  std::unordered_map<uint16_t, std::pair<uint16_t, uint16_t>> port_q_num;
  struct rte_mempool* pkt_len_buffer;
  struct rte_mempool* rx_burst_buffer;
  struct rte_mempool* rx_flow_id_buffer;
  struct rte_mempool* rx_metadata;
  struct rte_mempool* tx_metadata;
  uint64_t timestamp_mask_{0};
  uint64_t timestamp_offset_{0};
  std::array<struct rte_eth_conf, MAX_INTERFACES> local_port_conf;
  DpdkStats stats_;
  std::thread stats_thread_;
  int num_init = 0;
};

};  // namespace holoscan::advanced_network
