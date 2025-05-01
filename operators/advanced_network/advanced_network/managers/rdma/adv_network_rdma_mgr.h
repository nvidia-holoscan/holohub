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
#include "advanced_network/manager.h"
#include "advanced_network/common.h"
#include <mutex>

namespace holoscan::advanced_network {

struct rdma_qp_params {
   struct ibv_cq *rx_cq;
   struct ibv_cq *tx_cq;
   struct rte_ring *rx_ring;
   struct rte_ring *tx_ring;
   // mqd_t rx_mq;
   // mqd_t tx_mq;
   struct ibv_qp_init_attr qp_attr;
};

struct rdma_thread_params {
   bool active = false;
   struct rdma_cm_id *client_id;
   struct ibv_pd *pd;   
   rdma_qp_params qp_params;
   int if_idx;
   int queue_idx;
   bool ready_to_exit;
};

// Used to spawn a new server thread for a particular client
struct rdma_port_params {
   int if_idx;
   struct rdma_cm_id *server_id = nullptr;
   struct ibv_pd *pd = nullptr;
};

struct rdma_mr_params {
   MemoryRegionConfig params_;
   std::unordered_map<struct ibv_pd *, struct ibv_mr *> ctx_mr_map_;
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



class RdmaMgr : public Manager {
 public:
    static constexpr uint16_t DEFAULT_PORT = 18515;

    RdmaMgr() = default;
    ~RdmaMgr();
    bool set_config_and_initialize(const NetworkConfig &cfg) override;
    void initialize() override;
    void run() override;

    void* get_packet_ptr(BurstParams* burst, int idx) override;
    uint16_t get_packet_length(BurstParams* burst, int idx) override;
    uint16_t get_packet_flow_id(BurstParams* burst, int idx) override { return 0; }
    void* get_packet_extra_info(BurstParams* burst, int idx) override { return nullptr; }
    void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) override;
    uint16_t get_segment_packet_length(BurstParams* burst, int seg, int idx) override;
    Status get_tx_packet_burst(BurstParams *burst) override;
    Status set_eth_header(BurstParams *burst, int idx,
                                      char *dst_addr) override;
    Status set_ipv4_header(BurstParams *burst, int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) override;
    Status set_udp_header(BurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) override;
    Status set_udp_payload(BurstParams *burst, int idx,
                                      void *data, int len) override;
    bool is_tx_burst_available(BurstParams *burst) override;

    Status set_packet_lengths(BurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
    void free_all_segment_packets(BurstParams* burst, int seg) override {}
    void free_packet_segment(BurstParams* burst, int seg, int pkt) override { return; }
    void free_packet(BurstParams* burst, int pkt) override {}
    void free_all_packets(BurstParams* burst) override { return; }
    void free_rx_burst(BurstParams* burst) override;
    void free_tx_burst(BurstParams* burst) override;

    Status get_rx_burst(BurstParams **burst, int port, int q) override { return Status::NOT_SUPPORTED; };
    Status get_rx_burst(BurstParams **burst, uintptr_t conn_id, bool server) override;
    Status set_packet_tx_time(BurstParams *burst, int idx, uint64_t timestamp) override { return Status::NOT_SUPPORTED; };
    void free_rx_metadata(BurstParams *burst) override;
    void free_tx_metadata(BurstParams *burst) override;
    Status get_tx_metadata_buffer(BurstParams **burst) override;
    Status send_tx_burst(BurstParams *burst) override;
    uint64_t get_burst_tot_byte(BurstParams* burst) override;
    BurstParams* create_tx_burst_params() override;
    Status get_mac_addr(int port, char* mac) override { return Status::SUCCESS; }
    void shutdown() override;
    void print_stats() override;
    bool validate_config() const override { return true; }

    // RDMA-specific functions
    Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port, uintptr_t *conn_id) override;    
    Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port, const std::string& src_addr, uintptr_t *conn_id) override;
    Status rdma_get_port_queue(uintptr_t conn_id, uint16_t *port, uint16_t *queue) override;
    Status rdma_get_server_conn_id(const std::string& server_addr, uint16_t server_port, uint16_t queue_id, uintptr_t *conn_id) override;
    Status rdma_set_header(BurstParams* burst, RDMAOpCode op_code, uintptr_t conn_id, bool is_server, int num_pkts, uint64_t wr_id, const std::string& local_mr_name) override;
    RDMAOpCode rdma_get_opcode(BurstParams* burst) override;
    Status register_mr(std::string name, int intf, void *addr, size_t len, int flags);
    Status wait_on_key_xchg();
    void poll_cm_events();


 private:
    static constexpr int MAX_RDMA_CONNECTIONS = 128;
    static constexpr int MAX_CQ = 16;
    static constexpr int NUM_SGE_ELS = 1024;
    static constexpr int NUM_SGE_BUFS = 256;
    static constexpr int MAX_NUM_MR = 16; // Maximum number of memory registers to exchange
    static constexpr int MAX_OUSTANDING_WR = 64;
    static constexpr int MAX_NUM_PORTS = 4;
    static constexpr int MAX_RDMA_BATCH = 1024;

    bool initialized_ = false;
    std::unordered_map<struct rdma_cm_id *, std::vector<rdma_thread_params>> server_q_params_; 
    std::unordered_map<struct rdma_cm_id *, rdma_thread_params> client_q_params_;
    std::unordered_map<struct rdma_cm_id *, rdma_port_params> pd_params_;
    std::unordered_map<struct rdma_cm_id *, std::thread> worker_threads_;
    std::unordered_map<struct ibv_context *, struct ibv_pd *> pd_map_;
    std::unordered_map<std::string, rdma_mr_params> mrs_;
    std::unordered_map<std::string, rdma_remote_mr_info> remote_mrs_;
    std::unordered_map<std::string, struct rdma_cm_id*> server_str_to_id_;
    std::vector<rdma_key_xchg> lkey_mrs_;
    std::unordered_map<struct rdma_cm_id*, std::unordered_map<std::string, rdma_remote_mr_info>> endpoints_;
    std::queue<struct ibv_sge*> sge_bufs_;
    std::array<rdma_work_req, MAX_OUSTANDING_WR> out_wr_;
    uint64_t cur_wc_id_ = 0;
    std::queue<struct rte_ring*> server_rx_rings_;
    std::queue<struct rte_ring*> client_rx_rings_;
    struct rte_mempool* rx_meta;
    struct rte_mempool* tx_meta;
    struct rte_mempool* pkt_len_pool_;
    std::unordered_map<std::string, struct rte_mempool*> mr_pools_;
    std::queue<struct rte_ring*> server_tx_rings_;
    std::queue<struct rte_ring*> client_tx_rings_;
    std::unordered_map<struct rdma_cm_id*, struct rte_ring*> client_tx_rings_map_;
    std::unordered_map<struct rdma_cm_id*, struct rte_ring*> server_tx_rings_map_;
    std::unordered_map<struct rdma_cm_id*, struct rte_ring*> client_rx_rings_map_;
    std::unordered_map<struct rdma_cm_id*, struct rte_ring*> server_rx_rings_map_;
    std::unordered_map<std::string, struct rte_ring *> mem_pools_;
    struct rte_mempool* tx_burst_pool_;
    rdma_event_channel* cm_event_channel_;
    std::mutex threads_mutex_;
    std::mutex client_params_mutex_;
    std::thread main_thread_;
    void rdma_thread(bool is_server, rdma_thread_params *tparams);
    int setup_pools_and_rings();
    int rdma_register_mr(const MemoryRegionConfig &mr, void *ptr);
    int rdma_register_cfg_mrs();
    std::string generate_random_string(int len);
    static int set_affinity(int cpu_core);
    void init_client();
    void server_tx(int if_idx, int q);
    void server_rx(int if_idx, int q);
    bool ack_event(rdma_cm_event *cm_event);
    int mr_access_to_ibv(uint32_t access);
    bool get_ip_from_interface(const std::string_view &if_name, sockaddr_in &addr);
    int setup_thread_params(rdma_thread_params *params, bool is_server);
    int destroy_thread_params(rdma_thread_params *params);
    RDMAOpCode ibv_opcode_to_adv_net_opcode(ibv_wc_opcode opcode);
};

};  // namespace holoscan::advanced_network
