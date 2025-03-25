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
#include <atomic>
#include <thread>
#include <set>
#include <mqueue.h>
#include <unordered_map>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include "adv_network_mgr.h"
#include "adv_network_common.h"

namespace holoscan::ops {

struct rdma_qp_params {
   struct ibv_cq *rx_cq;
   struct ibv_cq *tx_cq;
   struct rte_ring *rx_ring;
   struct rte_ring *tx_ring;
   // mqd_t rx_mq;
   // mqd_t tx_mq;
   struct ibv_qp_init_attr qp_attr;
};


// Used to spawn a new server thread for a particular client
struct rdma_server_params {
   struct rdma_cm_id *client_id;
   struct ibv_pd *pd;
   struct ibv_comp_channel *iocc;
   struct ibv_cq *cq;
   std::vector<rdma_qp_params> qp_params;
};

struct rdma_mr_params {
   MemoryRegion params_;
   struct ibv_mr *mr_;
   void *ptr_;
};

struct rdma_remote_mr_info {
   std::string name;
   void *ptr;
   size_t len;
   uint32_t key;
};

struct rdma_key_xchg {
   uint64_t ptr;
   uint32_t key;
   size_t   size;
   char     name[32];
};

struct rdma_work_req {
   uint64_t wr_id;
   bool done = true;
   rdma_remote_mr_info mr;
};


class RdmaMgr : public ANOMgr {
 public:
    static constexpr uint16_t DEFAULT_PORT = 18515;

    RdmaMgr() = default;
    ~RdmaMgr();
    bool set_config_and_initialize(const AdvNetConfigYaml &cfg) override;
    void initialize() override;
    void run() override;

    void* get_pkt_ptr(AdvNetBurstParams* burst, int idx) override;
    uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx) override;
    uint16_t get_pkt_flow_id(AdvNetBurstParams* burst, int idx) override { return 0; }
    void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx) override { return nullptr; }
    void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) override;
    uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) override;
    AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams *burst) override;
    AdvNetStatus set_eth_hdr(AdvNetBurstParams *burst, int idx,
                                      char *dst_addr) override;
    AdvNetStatus set_ipv4_hdr(AdvNetBurstParams *burst, int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) override;
    AdvNetStatus set_udp_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) override;
    AdvNetStatus set_udp_payload(AdvNetBurstParams *burst, int idx,
                                      void *data, int len) override;
    bool tx_burst_available(AdvNetBurstParams *burst) override;

  AdvNetStatus set_pkt_lens(AdvNetBurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
    void free_all_seg_pkts(AdvNetBurstParams* burst, int seg) override {}
    void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) override {}
    void free_pkt(AdvNetBurstParams* burst, int pkt) override {}
    void free_all_pkts(AdvNetBurstParams* burst) override {}
    void free_rx_burst(AdvNetBurstParams* burst) override;
    void free_tx_burst(AdvNetBurstParams* burst) override;
    std::optional<uint16_t> get_port_from_ifname(const std::string &name) override;

    AdvNetStatus get_rx_burst(AdvNetBurstParams **burst) override;
    AdvNetStatus set_pkt_tx_time(AdvNetBurstParams *burst, int idx, uint64_t timestamp);
    void free_rx_meta(AdvNetBurstParams *burst) override;
    void free_tx_meta(AdvNetBurstParams *burst) override;
    AdvNetStatus get_tx_meta_buf(AdvNetBurstParams **burst) override;
    AdvNetStatus send_tx_burst(AdvNetBurstParams *burst) override;
    uint64_t get_burst_tot_byte(AdvNetBurstParams* burst) override;
    AdvNetBurstParams* create_burst_params() override;
    AdvNetStatus get_mac(int port, char* mac) override { return AdvNetStatus::SUCCESS; }
    int address_to_port(const std::string& addr) override { return 0; }
    void shutdown() override;
    void print_stats() override;
    bool validate_config() const override { return true; }

    // RDMA-specific functions
    AdvNetStatus rdma_connect_to_server(uint32_t server_addr, uint16_t server_port);    
    AdvNetStatus register_mr(std::string name, int intf, void *addr, size_t len, int flags);
    AdvNetStatus wait_on_key_xchg();
    void poll_cm_events();


 private:
    static constexpr int MAX_RDMA_CONNECTIONS = 8;
    static constexpr int MAX_CQ = 16;
    static constexpr int NUM_SGE_ELS = 1024;
    static constexpr int NUM_SGE_BUFS = 256;
    static constexpr int MAX_NUM_MR = 16; // Maximum number of memory registers to exchange
    static constexpr int MAX_OUSTANDING_WR = 64;
    std::vector<struct rdma_cm_id *> cm_server_id_;
    std::vector<struct rdma_cm_id *> cm_client_id_;
    std::vector<rdma_server_params> sparams_;
    std::vector<std::thread> txrx_workers;
    std::unordered_map<int, struct ibv_pd *> pd_map_;
    std::unordered_map<std::string, rdma_mr_params> mrs_;
    std::unordered_map<std::string, rdma_remote_mr_info> remote_mrs_;
    std::vector<rdma_key_xchg> lkey_mrs_;
    std::unordered_map<struct rdma_cm_id*, std::unordered_map<std::string, rdma_remote_mr_info>> endpoints_;
    std::queue<struct ibv_sge*> sge_bufs_;
    std::array<rdma_work_req, MAX_OUSTANDING_WR> out_wr_;
    uint64_t cur_wc_id_ = 0;
    struct rte_ring* rx_ring;
    struct rte_mempool* rx_meta;
    struct rte_mempool* tx_meta;
    std::unordered_map<uint32_t, struct rte_ring*> tx_rings;

    void rdma_thread(bool is_server, int if_idx, int q);
    int setup_pools_and_rings(int max_rx_batch, int max_tx_batch);
    int rdma_register_mr(const MemoryRegion &mr, void *ptr, int port_id);
    int rdma_register_cfg_mrs();
    std::string generate_random_string(int len);
    static int set_affinity(int cpu_core);
    int register_mrs();
    void init_client();
    void run_server();
    void run_client();
    void server_tx(int if_idx, int q);
    void server_rx(int if_idx, int q);
    bool ack_event(rdma_cm_event *cm_event);
    int mr_access_to_ibv(uint32_t access);
    bool get_ip_from_interface(const std::string_view &if_name, sockaddr_in &addr);
    int  setup_client_params_for_server(rdma_server_params *sparams, int if_idx);
};

};  // namespace holoscan::ops