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

class DpdkMgr : public ANOMgr {
 public:
    static_assert(MAX_INTERFACES <= RTE_MAX_ETHPORTS, "Too many interfaces configured");

    DpdkMgr() = default;
    ~DpdkMgr();
    void set_config_and_initialize(const AdvNetConfigYaml &cfg) override;
    void initialize() override;
    void run() override;
    static constexpr int JUMBFRAME_SIZE = 9100;
    static constexpr int DEFAULT_NUM_TX_BURST = 256;
    static constexpr int DEFAULT_NUM_RX_BURST = 64;
    uint16_t default_num_rx_desc = 8192;
    uint16_t default_num_tx_desc = 8192;
    int num_ports = 0;
    static constexpr int MAX_IFS = 4;
    static constexpr int num_lcores = 2;
    static constexpr int MEMPOOL_CACHE_SIZE = 32;
    static constexpr int MAX_PKT_BURST = 64;
    static constexpr uint32_t GPU_PAGE_SHIFT = 16;
    static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
    static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
    static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);
    static constexpr uint32_t CPU_PAGE_SIZE = 4096;
    static constexpr int BUFFER_SPLIT_SEGS = 2;
    static constexpr int MAX_ETH_HDR_SIZE = 18;

    void *get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx) override;
    void *get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx) override;
    uint16_t get_cpu_pkt_len(AdvNetBurstParams *burst, int idx) override;
    uint16_t get_gpu_pkt_len(AdvNetBurstParams *burst, int idx) override;
    AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams *burst) override;
    AdvNetStatus set_cpu_eth_hdr(AdvNetBurstParams *burst, int idx,
                                      uint8_t *dst_addr) override;
    AdvNetStatus set_cpu_ipv4_hdr(AdvNetBurstParams *burst, int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) override;
    AdvNetStatus set_cpu_udp_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) override;
    AdvNetStatus set_cpu_udp_payload(AdvNetBurstParams *burst, int idx,
                                      void *data, int len) override;
    bool tx_burst_available(AdvNetBurstParams *burst) override;

    AdvNetStatus set_pkt_len(AdvNetBurstParams *burst, int idx, int cpu_len, int gpu_len) override;
    void free_pkt(void *pkt) override;
    void free_pkts(void **pkts, int len) override;
    void free_rx_burst(AdvNetBurstParams *burst) override;
    void free_tx_burst(AdvNetBurstParams *burst) override;
    std::optional<uint16_t> get_port_from_ifname(const std::string &name) override;

    AdvNetStatus get_rx_burst(AdvNetBurstParams **burst) override;
    AdvNetStatus set_pkt_tx_time(AdvNetBurstParams *burst, int idx, uint64_t timestamp);
    void free_rx_meta(AdvNetBurstParams *burst) override;
    void free_tx_meta(AdvNetBurstParams *burst) override;
    AdvNetStatus get_tx_meta_buf(AdvNetBurstParams **burst) override;
    AdvNetStatus send_tx_burst(AdvNetBurstParams *burst) override;
    void shutdown() override;
    void print_stats() override;
    uint64_t get_burst_tot_byte(AdvNetBurstParams *burst) override;
    AdvNetBurstParams * create_burst_params() override;

 private:
    static std::string generate_random_string(int len);
    static int rx_core_worker(void *arg);
    static int tx_core_worker(void *arg);
    static void flush_packets(int port);
    void setup_accurate_send_scheduling_mask();
    int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
    struct rte_flow *add_flow(int port, const FlowConfig &cfg);
    std::optional<struct rte_pktmbuf_extmem> allocate_gpu_pktmbuf(int port_id, uint16_t pkt_size,
                                                                int num_mbufs, int gpu_dev);

    AdvNetConfigYaml cfg_;
    std::array<std::string, MAX_IFS> if_names;
    std::array<std::string, MAX_IFS> pcie_addrs;
    std::array<struct rte_ether_addr, MAX_IFS> mac_addrs;
    struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
    struct rte_ring *rx_ring;
    std::unordered_map<uint32_t, struct rte_ring *> tx_rings;
    std::unordered_map<uint32_t, struct rte_mempool *> tx_burst_buffers;
    std::unordered_map<uint32_t, struct rte_mempool *> tx_cpu_pkt_pools;
    std::unordered_map<uint32_t, struct rte_mempool *> tx_gpu_pkt_pools;
    std::unordered_map<uint32_t, struct rte_mempool *> rx_cpu_pkt_pools;
    std::unordered_map<uint32_t, struct rte_mempool *> rx_gpu_pkt_pools;
    struct rte_mempool *rx_burst_buffer;
    struct rte_mempool *rx_meta;
    struct rte_mempool *tx_meta;
    uint64_t timestamp_mask_{0};
    uint64_t timestamp_offset_{0};
    std::array<struct rte_eth_conf, MAX_INTERFACES> local_port_conf;

    int num_init = 0;
};

};  // namespace holoscan::ops
