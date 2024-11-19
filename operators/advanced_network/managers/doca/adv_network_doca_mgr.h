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

#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <tuple>
#include <unordered_map>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_flow.h>

#include <doca_log.h>
#include <doca_error.h>
#include <doca_version.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_mmap.h>
#include <doca_gpunetio.h>
#include <doca_dpdk.h>
#include <doca_flow.h>
#include <doca_dev.h>
#include <doca_buf_array.h>
#include <doca_pe.h>
#include <doca_eth_txq_gpu_data_path.h>

#include "adv_network_mgr.h"
#include "adv_network_common.h"

#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_SQ_DESCR_NUM 32768
#define CUDA_MAX_RX_NUM_PKTS 2048
#define CUDA_MAX_RX_TIMEOUT_NS 5000000  // 500us
#define CUDA_BLOCK_THREADS 512
#define MAX_DEFAULT_QUEUES 64
#define MAX_DEFAULT_SEM_X_QUEUE 512
#define MAX_TX_BURST 1024
#define THRESHOLD_PKT_SIZE 8192
#define THRESHOLD_BUF_NUM 32768

#define MPS_ENABLED 0
#define RX_PERSISTENT_ENABLED 1

struct adv_doca_rx_gpu_info {
  uint32_t num_pkts;
  uint32_t nbytes;
  uintptr_t gpu_pkt0_addr;
  uint32_t gpu_pkt0_idx;
};

static uint64_t next_power_of_two(uint64_t x) {
  x--;

  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;

  return x + 1;
}

namespace holoscan::ops {

static constexpr int JUMBFRAME_SIZE = 9100;
static constexpr int DEFAULT_NUM_TX_BURST = 256;
static constexpr int DEFAULT_NUM_RX_BURST = 64;
static constexpr int MAX_IFS = 4;
static constexpr int num_lcores = 2;
static constexpr int MEMPOOL_CACHE_SIZE = 32;
static constexpr uint32_t GPU_PAGE_SHIFT = 16;
static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
static constexpr uint32_t GPU_PAGE_OFFSET = (GPU_PAGE_SIZE - 1);
static constexpr uint32_t GPU_PAGE_MASK = (~GPU_PAGE_OFFSET);
static constexpr uint32_t CPU_PAGE_SIZE = 4096;
static constexpr int BUFFER_SPLIT_SEGS = 2;
static constexpr int MAX_ETH_HDR_SIZE = 18;
static constexpr int MAX_PORT_STR_LEN = 256;
static constexpr int MAX_PCIE_STR_LEN = 32;
static constexpr uint64_t default_flow_timeout_usec = 0;
static constexpr int TX_COMP_THRS = 2;

class DocaLogLevel {
 public:
  static std::string to_description_string(doca_log_level level) {
    auto it = level_to_string_description_map.find(level);
    if (it != level_to_string_description_map.end()) { return it->second; }
    throw std::logic_error(
        "Unrecognized log level, available options trace/debug/info/warn/error/critical/disable");
  }
  static doca_log_level from_ano_log_level(AnoLogLevel::Level ano_level) {
    auto it = ano_to_doca_log_level_map.find(ano_level);
    if (it != ano_to_doca_log_level_map.end()) { return it->second; }
    return DOCA_LOG_LEVEL_DISABLE;
  }

 private:
  static const std::unordered_map<doca_log_level, std::string> level_to_string_description_map;
  static const std::unordered_map<AnoLogLevel::Level, doca_log_level> ano_to_doca_log_level_map;
};

class DocaRxQueue {
 public:
  DocaRxQueue(struct doca_dev* dev, struct doca_gpu* gdev, struct doca_flow_port* df_port,
              uint16_t qid, int max_pkt_num, int max_pkt_size, enum doca_gpu_mem_type mtype);
  ~DocaRxQueue();
  doca_error_t create_udp_pipe(const FlowConfig& cfg, struct doca_flow_pipe* rxq_pipe_default);
  doca_error_t create_semaphore();
  doca_error_t destroy_semaphore();

  uint16_t qid;                         /* Number of queues */
  struct doca_gpu* gdev;                /* GPUNetio handler associated to queues*/
  struct doca_dev* ddev;                /* DOCA device handler associated to queues */
  struct doca_ctx* eth_rxq_ctx;         /* DOCA Ethernet receive queue context */
  struct doca_eth_rxq* eth_rxq_cpu;     /* DOCA Ethernet receive queue CPU handler */
  struct doca_gpu_eth_rxq* eth_rxq_gpu; /* DOCA Ethernet receive queue GPU handler */
  struct doca_mmap* pkt_buff_mmap;      /* DOCA mmap to receive packet with DOCA Ethernet queue */
  void* gpu_pkt_addr;                   /* DOCA mmap GPU memory address */
  void* cpu_pkt_addr;                   /* DOCA mmap CPU pinned memory address */
  int dmabuf_fd;                        /* GPU memory dmabuf file descriptor */
  int max_pkt_num;
  int max_pkt_size;
  struct doca_flow_port* df_port;              /* DOCA Flow port */
  struct doca_flow_pipe* rxq_pipe;             /* DOCA Flow receive pipe */
  struct doca_flow_pipe_entry* root_udp_entry; /* DOCA Flow root entry */

  struct doca_gpu_semaphore* sem_cpu;     /* One semaphore per queue to report stats, CPU handler*/
  struct doca_gpu_semaphore_gpu* sem_gpu; /* One semaphore per queue to report stats, GPU handler*/
  enum doca_gpu_mem_type mtype;
};

class DocaTxQueue {
 public:
  DocaTxQueue(struct doca_dev* dev, struct doca_gpu* gdev, uint16_t qid, int max_pkt_num,
              int max_pkt_size, enum doca_gpu_mem_type mtype,
              doca_eth_txq_gpu_event_notify_send_packet_cb_t event_notify_send_packet_cb);
  ~DocaTxQueue();

  uint16_t qid;                         /* Number of queues */
  struct doca_gpu* gdev;                /* GPUNetio handler associated to queues*/
  struct doca_dev* ddev;                /* DOCA device handler associated to queues */
  struct doca_ctx* eth_txq_ctx;         /* DOCA Ethernet send queue context */
  struct doca_eth_txq* eth_txq_cpu;     /* DOCA Ethernet send queue CPU handler */
  struct doca_gpu_eth_txq* eth_txq_gpu; /* DOCA Ethernet send queue GPU handler */
  struct doca_mmap* pkt_buff_mmap;      /* DOCA mmap to receive packet with DOCA Ethernet queue */
  void* gpu_pkt_addr;                   /* DOCA mmap GPU memory address */
  void* cpu_pkt_addr;                   /* DOCA mmap CPU pinned memory address */
  int dmabuf_fd;                        /* GPU memory dmabuf file descriptor */
  int max_pkt_num;
  int max_pkt_size;
  struct doca_buf_arr* buf_arr;         /* DOCA buffer array object around GPU memory buffer */
  struct doca_gpu_buf_arr* buf_arr_gpu; /* DOCA buffer array GPU handle */
  std::atomic<uint32_t> buff_arr_idx;
  struct doca_pe* pe;
  std::atomic<uint32_t> tx_cmp_posted;
  enum doca_gpu_mem_type mtype;
};

class DocaMgr : public ANOMgr {
 public:
  static_assert(MAX_INTERFACES <= RTE_MAX_ETHPORTS, "Too many interfaces configured");
  DocaMgr() = default;
  ~DocaMgr();
  bool set_config_and_initialize(const AdvNetConfigYaml& cfg) override;
  void initialize() override;
  void run() override;
  // int SetupPoolsAndRings();
  static int rx_core(void* arg);
  static int tx_core(void* arg);
  uint16_t default_num_rx_desc = 8192;
  uint16_t default_num_tx_desc = 8192;
  int num_ports = 0;

  void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) override;
  void* get_pkt_ptr(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) override;
  void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_pkt_flow_id(AdvNetBurstParams* burst, int idx) override;
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
  void free_all_seg_pkts(AdvNetBurstParams* burst, int seg) override{};
  void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) override{};
  void free_pkt(AdvNetBurstParams* burst, int pkt) override{};
  void free_all_pkts(AdvNetBurstParams* burst) override{};
  void free_rx_burst(AdvNetBurstParams* burst) override;
  void free_tx_burst(AdvNetBurstParams* burst) override;
  std::optional<uint16_t> get_port_from_ifname(const std::string& name) override;

  AdvNetStatus get_rx_burst(AdvNetBurstParams** burst) override;
  AdvNetStatus set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_meta(AdvNetBurstParams* burst) override;
  void free_tx_meta(AdvNetBurstParams* burst) override;
  AdvNetStatus get_tx_meta_buf(AdvNetBurstParams** burst) override;
  AdvNetStatus send_tx_burst(AdvNetBurstParams* burst) override;
  AdvNetStatus get_mac(int port, char* mac) override;
  int address_to_port(const std::string& addr) override;
  void shutdown() override;
  void print_stats() override;
  bool validate_config() const override;

  uint64_t get_burst_tot_byte(AdvNetBurstParams* burst) override;
  AdvNetBurstParams* create_burst_params() override;

 private:
  doca_error_t init_doca_devices();
  doca_error_t create_root_pipe(int port_id);
  doca_error_t create_default_pipe(int port_id, uint32_t cnt_defq);
  struct doca_flow_port* init_doca_flow(uint16_t port_id, uint8_t rxq_num);
  int setup_pools_and_rings(int max_tx_batch);
  std::string GetQueueName(int port, int q, AdvNetDirection dir);
  std::unordered_map<uint32_t, struct rte_ring*> tx_rings;
  struct rte_ring* rx_ring;
  struct rte_mempool* rx_meta;
  struct rte_mempool* tx_meta;
  std::unordered_map<uint32_t, DocaRxQueue*> rx_q_map_;
  std::unordered_map<uint32_t, DocaTxQueue*> tx_q_map_;
  std::array<struct rte_eth_conf, MAX_INTERFACES> local_port_conf;
  std::array<struct rte_ether_addr, MAX_IFS> mac_addrs;

  uint16_t dpdk_port_id;
  std::string net_bdf;
  std::array<struct doca_dev*, MAX_IFS> ddev{nullptr};
  std::array<struct doca_gpu*, MAX_GPUS> gdev{nullptr};
  std::array<struct doca_flow_port*, MAX_IFS> df_port;
  std::array<struct doca_flow_pipe*, MAX_IFS> root_pipe;
  struct doca_flow_pipe_entry* root_udp_entry;
  uint16_t rxq_num;
  uint16_t txq_num;

  bool initialized = false;
  int num_init = 0;
  struct doca_flow_pipe* rxq_pipe_default;             /* DOCA Flow receive pipe default */
  struct doca_flow_pipe_entry* root_udp_entry_default; /* DOCA Flow root entry */
  AdvNetBurstParams burst[MAX_TX_BURST];
  std::atomic<uint32_t> burst_tx_idx;

  std::thread worker_th[16];
  int worker_th_idx;
  std::set<int> gpu_mr_devs;
};

extern DocaMgr doca_mgr;
};  // namespace holoscan::ops
