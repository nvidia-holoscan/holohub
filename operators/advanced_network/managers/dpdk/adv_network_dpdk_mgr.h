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
#include "adv_network_mgr.h"
#include "adv_network_common.h"

namespace holoscan::ops {

struct DPDKQueueConfig {
  std::vector<struct rte_mempool*> pools;
  struct rte_eth_rxconf rxconf_qsplit;
  std::vector<union rte_eth_rxseg> rx_useg;
};

class DpdkLogLevel {
 public:
  enum Level {
    OFF = 0,
    EMERGENCY = 1,
    ALERT = 2,
    CRITICAL = 3,
    ERROR = 4,
    WARN = 5,
    NOTICE = 6,
    INFO = 7,
    DEBUG = 8,
  };

  static std::string to_description_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<0>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }

  static std::string to_cmd_string(Level level) {
    auto it = level_to_cmd_map.find(level);
    if (it != level_to_cmd_map.end()) { return std::get<1>(it->second); }
    throw std::logic_error(
        "Unrecognized log level, available options "
        "debug/info/notice/warn/error/critical/alert/emergency/off");
  }

  static Level from_ano_log_level(AnoLogLevel::Level ano_level) {
    auto it = ano_to_dpdk_log_level_map.find(ano_level);
    if (it != ano_to_dpdk_log_level_map.end()) { return it->second; }
    return OFF;
  }

 private:
  /**
   * A map of log level to a tuple of the description and command strings.
   */
  static const std::unordered_map<Level, std::tuple<std::string, std::string>> level_to_cmd_map;
  static const std::unordered_map<AnoLogLevel::Level, Level> ano_to_dpdk_log_level_map;
};

/**
 * @class DpdkLogLevelCommandBuilder
 * @brief Concrete class for building DPDK log level commands.
 *
 * This class implements the ManagerLogLevelCommandBuilder interface to provide
 * specific command flag strings for managing DPDK log levels.
 */
class DpdkLogLevelCommandBuilder : public ManagerLogLevelCommandBuilder {
 public:
  /**
   * @brief Constructor for DpdkLogLevelCommandBuilder.
   *
   * @param ano_level The log level from AnoLogLevel to be converted to DPDK log level.
   */
  explicit DpdkLogLevelCommandBuilder(AnoLogLevel::Level ano_level)
      : level_(DpdkLogLevel::from_ano_log_level(ano_level)) {}

  /**
   * @brief Get the command flag strings for DPDK log levels.
   *
   * This function returns the specific command flag strings required to set
   * the DPDK log levels.
   *
   * @return A vector of command flag strings.
   */
  std::vector<std::string> get_cmd_flags_strings() const override {
    return {"--log-level=" + DpdkLogLevel::to_cmd_string(DpdkLogLevel::Level::OFF),
            "--log-level=pmd.net.mlx5:" + DpdkLogLevel::to_cmd_string(level_)};
  }

 private:
  DpdkLogLevel::Level level_;  ///< The DPDK log level.
};

class DpdkMgr : public ANOMgr {
 public:
  static_assert(MAX_INTERFACES <= RTE_MAX_ETHPORTS, "Too many interfaces configured");

  DpdkMgr() = default;
  ~DpdkMgr();
  bool set_config_and_initialize(const AdvNetConfigYaml& cfg) override;
  void initialize() override;
  void run() override;
  static constexpr int JUMBOFRAME_SIZE = 9100;
  static constexpr int DEFAULT_NUM_TX_BURST = 256;
  static constexpr int DEFAULT_NUM_RX_BURST = 64;
  uint16_t default_num_rx_desc = 8192;
  uint16_t default_num_tx_desc = 8192;
  int num_ports = 0;
  static constexpr int num_lcores = 2;
  static constexpr int MEMPOOL_CACHE_SIZE = 32;
  static constexpr int MAX_PKT_BURST = 64;

  static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
  static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);
  static constexpr uint32_t CPU_PAGE_SIZE = 4096;
  static constexpr int BUFFER_SPLIT_SEGS = 2;
  static constexpr int MAX_ETH_HDR_SIZE = 18;

  void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) override;
  void* get_pkt_ptr(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) override;
  uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_pkt_flow_id(AdvNetBurstParams* burst, int idx) override;
  void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx) override;
  AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams* burst) override;
  AdvNetStatus set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) override;
  AdvNetStatus set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                            unsigned int src_host, unsigned int dst_host) override;
  AdvNetStatus set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                           uint16_t dst_port) override;
  AdvNetStatus set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) override;
  bool tx_burst_available(AdvNetBurstParams* burst) override;

  AdvNetStatus set_pkt_lens(AdvNetBurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  void free_all_seg_pkts(AdvNetBurstParams* burst, int seg) override;
  void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) override;
  void free_pkt(AdvNetBurstParams* burst, int pkt) override;
  void free_all_pkts(AdvNetBurstParams* burst) override;
  void free_rx_burst(AdvNetBurstParams* burst) override;
  void free_tx_burst(AdvNetBurstParams* burst) override;
  std::optional<uint16_t> get_port_from_ifname(const std::string& name) override;

  AdvNetStatus get_rx_burst(AdvNetBurstParams** burst) override;
  AdvNetStatus set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_meta(AdvNetBurstParams* burst) override;
  void free_tx_meta(AdvNetBurstParams* burst) override;
  AdvNetStatus get_tx_meta_buf(AdvNetBurstParams** burst) override;
  AdvNetStatus send_tx_burst(AdvNetBurstParams* burst) override;
  int address_to_port(const std::string& addr) override;
  AdvNetStatus get_mac(int port, char* mac) override;
  void shutdown() override;
  void print_stats() override;
  void adjust_memory_regions() override;
  uint64_t get_burst_tot_byte(AdvNetBurstParams* burst) override;
  AdvNetBurstParams* create_burst_params() override;
  bool validate_config() const override;

 private:
  static void PrintDpdkStats(int port);
  static std::string generate_random_string(int len);
  static int rx_core_worker(void* arg);
  static int tx_core_worker(void* arg);
  static void flush_packets(int port);
  void setup_accurate_send_scheduling_mask();
  int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
  struct rte_flow* add_flow(int port, const FlowConfig& cfg);
  AdvNetStatus register_mrs();
  AdvNetStatus map_mrs();
  void create_dummy_rx_q();
  int numa_from_mem(const MemoryRegion& mr);
  struct rte_flow* add_modify_flow_set(int port, int queue, const char* buf, int len,
                                       AdvNetDirection direction);

  void apply_tx_offloads(int port);

  std::array<std::string, MAX_IFS> if_names;
  std::array<std::string, MAX_IFS> pcie_addrs;
  std::array<struct rte_ether_addr, MAX_IFS> mac_addrs;
  struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
  struct rte_ring* rx_ring;
  std::unordered_map<uint32_t, struct rte_ring*> tx_rings;
  std::unordered_map<uint32_t, struct rte_mempool*> tx_burst_buffers;
  std::unordered_map<std::string, std::shared_ptr<struct rte_pktmbuf_extmem>> ext_pktmbufs_;
  std::unordered_map<uint32_t, DPDKQueueConfig*> rx_q_map_;
  std::unordered_map<uint32_t, DPDKQueueConfig*> tx_q_map_;
  struct rte_mempool* pkt_len_buffer;
  struct rte_mempool* rx_burst_buffer;
  struct rte_mempool* rx_flow_id_buffer;
  struct rte_mempool* rx_meta;
  struct rte_mempool* tx_meta;
  uint64_t timestamp_mask_{0};
  uint64_t timestamp_offset_{0};
  std::array<struct rte_eth_conf, MAX_INTERFACES> local_port_conf;

  int num_init = 0;
};

};  // namespace holoscan::ops
