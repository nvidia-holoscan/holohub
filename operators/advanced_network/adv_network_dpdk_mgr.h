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
#include "adv_network_common.h"

namespace holoscan::ops {

class DpdkMgr {
 public:
    DpdkMgr() {
      static_assert(MAX_INTERFACES <= RTE_MAX_ETHPORTS, "Too many interfaces configured");
    }
    ~DpdkMgr();
    void SetConfigAndInitialize(const AdvNetConfigYaml &cfg);
    int GetRxPkts(void **pkts, int num);
    void Initialize();
    void Run();
    void wait();
    static int rx_core(void *arg);
    static int tx_core(void *arg);
    static void check_pkts_to_free(rte_ring *msg_ring,
          rte_mempool *burst_pool, rte_mempool *meta_pool);
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


 private:
    static void flush_packets(int port);
    int SetupPoolsAndRings(int max_rx_batch, int max_tx_batch);
    struct rte_flow *AddFlow(int port, const FlowConfig &cfg);
    std::string GetQueueName(int port, int q, AdvNetDirection dir);
    std::optional<struct rte_pktmbuf_extmem> AllocateGpuPktMbuf(int port_id, uint16_t pkt_size,
                                                                int num_mbufs, int gpu_dev);

    AdvNetConfigYaml cfg_;
    std::array<std::string, MAX_IFS> if_names;
    std::array<std::string, MAX_IFS> pcie_addrs;
    std::array<struct rte_ether_addr, MAX_IFS> mac_addrs;
    struct rte_ether_addr conf_ports_eth_addr[RTE_MAX_ETHPORTS];
    struct rte_ring *rx_ring;
    std::unordered_map<uint32_t, struct rte_ring *> tx_rings;
    std::unordered_map<uint32_t, struct rte_mempool *> tx_burst_buffers;
    struct rte_mempool *rx_burst_buffer;
    struct rte_mempool *rx_meta;
    struct rte_mempool *tx_meta;
    std::array<struct rte_eth_conf, MAX_INTERFACES> local_port_conf;

    bool initialized = false;
    int num_init = 0;
};

extern DpdkMgr dpdk_mgr;
};  // namespace holoscan::ops
