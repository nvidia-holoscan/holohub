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

#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <mqueue.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_errno.h>
#include "../dpdk/adv_network_dpdk_log.h"
#include "adv_network_rdma_mgr.h"

/* The ordering of most RDMA/CM setup follows the ordering specified here:
   https://man7.org/linux/man-pages/man7/rdma_cm.7.html
   The exception is that there is no standard way to pass around keys, so we use standard 
   sends and receives.
*/

namespace holoscan::advanced_network {
  std::atomic<bool> rdma_force_quit = false;

  bool RdmaMgr::set_config_and_initialize(const NetworkConfig &cfg) {
    if (!this->initialized_) {
      cfg_ = cfg;
      initialize();
    }    

    return true;
  }

  // Common ANO functions
  Status RdmaMgr::set_packet_lengths(BurstParams* burst, int idx,
                                    const std::initializer_list<int>& lens) {
// populate later
    return Status::SUCCESS;
  }  

  void* RdmaMgr::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
    return burst->pkts[seg][idx];
  }

  void* RdmaMgr::get_packet_ptr(BurstParams* burst, int idx) {
    return burst->pkts[0][idx];
  }  
  
  uint16_t RdmaMgr::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
    return burst->pkt_lens[seg][idx];
  }

  uint16_t RdmaMgr::get_packet_length(BurstParams* burst, int idx) {
    return burst->pkt_lens[0][idx];
  }

  Status RdmaMgr::set_eth_header(BurstParams *burst, int idx,
                                    char *dst_addr) {
    HOLOSCAN_LOG_CRITICAL("Cannot set Ethernet header in RDMA mode");
    return Status::NOT_SUPPORTED;
  }

  Status RdmaMgr::set_ipv4_header(BurstParams *burst, int idx,
                                    int ip_len,
                                    uint8_t proto,
                                    unsigned int src_host,
                                    unsigned int dst_host) {
    HOLOSCAN_LOG_CRITICAL("Cannot set IPv4 header in RDMA mode");
    return Status::NOT_SUPPORTED;
  }

  Status RdmaMgr::set_udp_header(BurstParams *burst,
                                    int idx,
                                    int udp_len,
                                    uint16_t src_port,
                                    uint16_t dst_port) {
    HOLOSCAN_LOG_CRITICAL("Cannot set UDP header in RDMA mode");
    return Status::NOT_SUPPORTED;
  }

  Status RdmaMgr::set_udp_payload(BurstParams *burst, int idx,
                                    void *data, int len) {
    HOLOSCAN_LOG_CRITICAL("Cannot set UDP payload in RDMA mode");
    return Status::NOT_SUPPORTED;
  }  

  uint64_t RdmaMgr::get_burst_tot_byte(BurstParams* burst) {
    return 0;
  }

  BurstParams* RdmaMgr::create_tx_burst_params() {
    return new BurstParams();
  }  

  bool RdmaMgr::get_ip_from_interface(const std::string_view &if_name, sockaddr_in &addr) {
    struct ifaddrs *ifaddr, *ifa;
    bool found = false;

    // Initialize the sockaddr_in structure
    memset(&addr, 0, sizeof(sockaddr_in));
    addr.sin_family = AF_INET;

    // Get the list of network interfaces
    if (getifaddrs(&ifaddr) == -1) {
        HOLOSCAN_LOG_CRITICAL("Failed to get a list of interfaces");
        return false;
    }

    // Loop through the list of interfaces
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Interface {} has no address", ifa->ifa_name);
        continue;
      }

      if (ifa->ifa_addr->sa_family != AF_INET) {
        continue; // Only IPv4 for now
      }

      // Check if the interface name matches
      if (if_name == ifa->ifa_name) {
          struct sockaddr_in *in_addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
          addr.sin_addr = in_addr->sin_addr;

          found = true;
          break;
      }
    }

    freeifaddrs(ifaddr);
    return found;
  }

  inline bool RdmaMgr::ack_event(rdma_cm_event *cm_event) {
    int ret = rdma_ack_cm_event(cm_event);
    if (ret != 0) {
      HOLOSCAN_LOG_ERROR("Failed to acknowledge CM event: {}", ret);
      return false;
    }
    return true;
  }

  int RdmaMgr::mr_access_to_ibv(uint32_t access) {
    int ibv_access = 0;

    if (access & MEM_ACCESS_LOCAL) {
      ibv_access |= IBV_ACCESS_LOCAL_WRITE;
    }
    if (access & MEM_ACCESS_RDMA_READ) {
      ibv_access |= IBV_ACCESS_REMOTE_READ ;
    }
    if (access & MEM_ACCESS_RDMA_WRITE) {
      ibv_access |= IBV_ACCESS_REMOTE_WRITE;
    }

    return ibv_access;
  }

  int RdmaMgr::rdma_register_mr(const MemoryRegionConfig &mr, void *ptr) {
    rdma_mr_params params;
    params.params_ = mr;
    params.ptr_ = ptr;

    // For now register the MR with every PD we have
    for (const auto &pd: pd_map_) {
      for (const auto &pd: pd.second) {
        if (pd != nullptr) {
          int access = mr_access_to_ibv(mr.access_);
          params.mr_ = ibv_reg_mr(pd, ptr, mr.buf_size_ * mr.num_bufs_, access);
          if (params.mr_ == nullptr) {
            HOLOSCAN_LOG_CRITICAL("Failed to register MR {} on PD {}", mr.name_, (void*)pd);
            return -1;
          }
          else {
            HOLOSCAN_LOG_INFO("Successfully registered MR {} with {} bytes on PD {}", 
              mr.name_, mr.buf_size_ * mr.num_bufs_, (void*)pd);
          }
        }
      }
    }

    mrs_[mr.name_] = params;  

    return 0;  
  }

  int RdmaMgr::rdma_register_cfg_mrs() {
    HOLOSCAN_LOG_INFO("Registering memory regions");

    if (allocate_memory_regions() != Status::SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to allocate memory");
      return -1;
    } 

    for (const auto &mr: cfg_.mrs_) {
      int ret = rdma_register_mr(mr.second, ar_[mr.second.name_].ptr_);
      if (ret < 0) {
        return ret;
      }
    }

    // Register all the MRs for exchanging keys
    for (int p = 0; p < cfg_.ifs_.size(); p++) {
      const std::string name = std::string("lkey_") + std::to_string(p);
      MemoryRegionConfig mr{
        name,
        cfg_.ifs_[p].name_,
        MemoryKind::HOST,
        MEM_ACCESS_LOCAL,
        sizeof(rdma_key_xchg),
        MAX_NUM_MR
      };

      lkey_mrs_.emplace_back(rdma_key_xchg{}); // Dummy entry where data will be written

      if (rdma_register_mr(mr, &lkey_mrs_[p]) < 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to register key exchange MR for interface {}", p);
        return -1;
      }
    }

    return 0;
  }

  Status RdmaMgr::wait_on_key_xchg() {
    return Status::SUCCESS;
  }

  /**
   * Set up all parameters needed for a newly-connected client
  */
  int RdmaMgr::setup_thread_params(rdma_thread_params *params) {
    // RX/TX queues should be symmetric with RDMA
    rdma_qp_params qp_params;
    const int port_id = cfg_.ifs_[static_cast<int>(params->if_idx)].port_id_;

    qp_params.rx_cq = ibv_create_cq(params->client_id->verbs, 
      MAX_CQ, nullptr, nullptr,	0);
    if (qp_params.rx_cq == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to create RX queue pair! {}", errno);
      return -1;
    }  

    qp_params.tx_cq = ibv_create_cq(params->client_id->verbs, 
      MAX_CQ, nullptr, nullptr,	0);
    if (qp_params.tx_cq == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to create TX queue pair! {}", errno);
      return -1;
    }              

    memset(&qp_params.qp_attr, 0, sizeof(qp_params.qp_attr));       
    qp_params.qp_attr.cap.max_recv_sge = 1; // No header-data split in RDMA right now
    qp_params.qp_attr.cap.max_recv_wr = MAX_OUSTANDING_WR;
    qp_params.qp_attr.cap.max_send_sge = 1;
    qp_params.qp_attr.cap.max_send_wr = MAX_OUSTANDING_WR;

    if (cfg_.ifs_[static_cast<int>(params->if_idx)].rdma_.xmode_ == RDMATransportMode::RC) {
      qp_params.qp_attr.qp_type = IBV_QPT_RC;
    }
    else if (cfg_.ifs_[static_cast<int>(params->if_idx)].rdma_.xmode_ == RDMATransportMode::UC) {
      qp_params.qp_attr.qp_type = IBV_QPT_UC;
    }
    else {
      HOLOSCAN_LOG_ERROR("RDMA transport mode {} not supported!", 
            static_cast<int>(cfg_.ifs_[static_cast<int>(params->if_idx)].rdma_.xmode_));
      return -1;
    }

    // Share the CQ between TX and RX
    qp_params.qp_attr.recv_cq = qp_params.rx_cq;
    qp_params.qp_attr.send_cq = qp_params.tx_cq;

    int ret = rdma_create_qp(params->client_id, params->pd, &qp_params.qp_attr);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to create QP: {}", errno);
      return -1;
    } 

    // Create POSIX message queues for talking to client
    struct mq_attr attr;

    attr.mq_flags = 0;
    attr.mq_maxmsg = 128;
    attr.mq_msgsize = sizeof(BurstParams);
    attr.mq_curmsgs = 0;

    qp_params.rx_ring = rx_ring;  

    const uint32_t key = (port_id << 16) | static_cast<int>(params->queue_idx);
    qp_params.tx_ring = tx_rings[key];

    params->qp_params = qp_params;

    return 0;
  }

  inline int RdmaMgr::set_affinity(int cpu_core) {
    // Set the CPU affinity of our thread
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to set thread affinity to core {}", cpu_core);
      return -1;
    }

    return 0;    
  }
  
  /**
   * Worker thread for a client or server. Each thread handles one queue pair.
  */
  void RdmaMgr::rdma_thread(bool is_server, rdma_thread_params tparams) {
    struct ibv_wc wc;
    int num_comp;
    const auto &qref = cfg_.ifs_[tparams.if_idx].tx_.queues_[tparams.queue_idx];
    const long cpu_core = strtol(qref.common_.cpu_core_.c_str(), NULL, 10);
    struct rte_ring *tx_ring = tparams.qp_params.tx_ring;

    if (set_affinity(cpu_core) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to set RDMA core affinity");
      return;
    }

    HOLOSCAN_LOG_INFO("Affined {} RDMA thread to core {}", is_server ? "Server" : "Client", cpu_core);

    // Main TX loop. Wait for send requests from the transmitters to arrive for sending. Also
    // periodically poll the CQ.
    while (!rdma_force_quit.load()) {
      // Check RQ first to reduce latency
      while ((num_comp = ibv_poll_cq(tparams.qp_params.rx_cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", (int)wc.status, (int64_t)wc.wr_id, (int)wc.opcode);
        }
        else {
          HOLOSCAN_LOG_INFO("Received message from client: {}", (int)wc.status);
        }
      }

      // Check TX CQ for completion
      while ((num_comp = ibv_poll_cq(tparams.qp_params.tx_cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", (int)wc.status, (int64_t)wc.wr_id, (int)wc.opcode);
          continue;
        }

        if (wc.opcode == IBV_WC_RDMA_READ) {
          // BurstParams read_burst;
          // read_burst.rdma_hdr.

          // out_wr_[cur_wc_id_].wr_id = cur_wc_id_;
          // out_wr_[cur_wc_id_].done  = false;
          // out_wr_[cur_wc_id_].mr = *(endpt->second);
          // out_wr_[cur_wc_id_].mr.ptr = burst.cpu_pkts[p];
          // cur_wc_id_++;

          // auto ret = mq_send(tparams.rx_mq, &read_burst, sizeof(read_burst), 10);
          // if (ret != 0) {
          //   HOLOSCAN_LOG_CRITICAL("Failed to send to message queue: {}", errno);
          //   continue;
          // }
        }
      }

      // Now handle any incoming messages
      BurstParams burst;

      //ssize_t bytes = mq_receive(tparams.tx_mq, reinterpret_cast<char*>(&burst), sizeof(burst), nullptr);
      if (rte_ring_dequeue(tparams.qp_params.tx_ring, reinterpret_cast<void**>(&burst)) != 0) { 
        continue; 
      }

      const auto endpt = endpoints_.find(reinterpret_cast<struct rdma_cm_id*>(burst.rdma_hdr.dst_key));
      if (endpt == endpoints_.end()) {
        HOLOSCAN_LOG_ERROR("Trying to send to client {}, but that client is not connected", burst.rdma_hdr.dst_key);
        continue;
      }

      const auto local_mr = mrs_.find(burst.rdma_hdr.local_mr_name);
      if (local_mr == mrs_.end()) {
        HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry", burst.rdma_hdr.local_mr_name);
        free_tx_burst(&burst);
        continue;
      }

      switch (burst.rdma_hdr.opcode) {
        case AdvNetRDMAOpCode::SEND:
        { // perform send operation
          // Currently we expect SEND operations to be rare, and certainly not with thousands of
          // packets. For that reason we post one at a time and do not try to batch.
          for (int p = 0; p < burst.rdma_hdr.num_pkts; p++) {
            ibv_send_wr wr;
            ibv_send_wr *bad_wr;
            ibv_sge sge;

            memset(&wr, 0, sizeof(wr));
            sge.addr      = (uint64_t)burst.pkts[0][p];
            sge.length    = (uint32_t)burst.pkt_lens[0][p];
            sge.lkey      = local_mr->second.mr_->lkey;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            wr.opcode     = IBV_WR_SEND;
            wr.send_flags = IBV_SEND_SIGNALED;

            int ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
            if (ret != 0) {
              HOLOSCAN_LOG_CRITICAL("Failed to post SEND request, errno: {}", errno);
              free_tx_burst(&burst);
              continue;
            }   
            }

          break; 
        }
        case AdvNetRDMAOpCode::RDMA_WRITE: [[fall_through]];
        case AdvNetRDMAOpCode::RDMA_WRITE_IMM:
        {
          for (int p = 0; p < burst.rdma_hdr.num_pkts; p++) {
            ibv_send_wr wr;
            ibv_send_wr *bad_wr;
            ibv_sge sge;

            memset(&wr, 0, sizeof(wr));
            sge.addr      = (uint64_t)burst.pkts[0][p];
            sge.length    = (uint32_t)burst.pkt_lens[0][p];
            sge.lkey      = local_mr->second.mr_->lkey;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            if (burst.rdma_hdr.opcode==AdvNetRDMAOpCode::RDMA_WRITE) {
              wr.opcode   = IBV_WR_RDMA_WRITE;
            }
            else {
              wr.opcode   = IBV_WR_RDMA_WRITE_WITH_IMM;
              wr.imm_data = htonl(burst.rdma_hdr.imm);
            }

            wr.send_flags = IBV_SEND_SIGNALED;

            // Look up remote key
            const auto remote_mr = endpt->second.find(burst.rdma_hdr.remote_mr_name);
            if (remote_mr == endpt->second.end()) {
              HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry for client {}", 
                    burst.rdma_hdr.remote_mr_name, burst.rdma_hdr.dst_key);
              free_tx_burst(&burst);
              continue;
            }              
            wr.wr.rdma.rkey = remote_mr->second.key;
            wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(burst.rdma_hdr.raddr);

            int ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
            if (ret != 0) {
              HOLOSCAN_LOG_CRITICAL("Failed to post SEND request, errno: {}", errno);
              free_tx_burst(&burst);
              continue;
            }
          }
          break;
        }
        case AdvNetRDMAOpCode::RDMA_READ:
        {
          for (int p = 0; p < burst.rdma_hdr.num_pkts; p++) {
            ibv_send_wr wr;
            ibv_send_wr *bad_wr;
            ibv_sge sge;

            memset(&wr, 0, sizeof(wr));
            sge.addr      = (uint64_t)burst.pkts[0][p];
            sge.length    = (uint32_t)burst.pkt_lens[0][p];
            sge.lkey      = local_mr->second.mr_->lkey;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            wr.opcode     = IBV_WR_RDMA_READ;
            wr.send_flags = IBV_SEND_SIGNALED;

            // Look up remote key
            const auto remote_mr = endpt->second.find(burst.rdma_hdr.remote_mr_name);
            if (remote_mr == endpt->second.end()) {
              HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry for client {}", 
                    burst.rdma_hdr.remote_mr_name, burst.rdma_hdr.dst_key);
              free_tx_burst(&burst);
              continue;
            }              
            wr.wr.rdma.rkey  = remote_mr->second.key;
            wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(burst.rdma_hdr.raddr);

            int ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
            if (ret != 0) {
              HOLOSCAN_LOG_CRITICAL("Failed to post SEND request, errno: {}", errno);
              free_tx_burst(&burst);
              continue;
            }

            out_wr_[cur_wc_id_].wr_id = cur_wc_id_;
            out_wr_[cur_wc_id_].done  = false;
            out_wr_[cur_wc_id_].mr = remote_mr->second;
            out_wr_[cur_wc_id_].mr.ptr = burst.pkts[0][p];
            cur_wc_id_++;
          }
          break;
        }          
      }
    }
  }

  Status RdmaMgr::rdma_connect_to_server(const std::string &server_addr_str, uint16_t server_port, uintptr_t *conn_id) {
    struct sockaddr_in addr;
    struct rdma_cm_id *cm_id = nullptr;
    struct rdma_event_channel *ec = nullptr;
    struct rdma_cm_event *event = nullptr;
    struct rdma_conn_param conn_param = {};
    uint32_t server_addr;

    // Convert IPv4 address from string to int32
    if (inet_pton(AF_INET, server_addr_str.c_str(), &server_addr) != 1) {
      HOLOSCAN_LOG_CRITICAL("Failed to convert IP address: {}", server_addr_str);
      return Status::CONNECT_FAILURE;
    }

    *conn_id = 0;

    ec = rdma_create_event_channel();    
    if (ec == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to create a CM channel: {}", errno);
      return Status::CONNECT_FAILURE;
    }

    // Create RDMA id
    if (rdma_create_id(ec, &cm_id, nullptr, RDMA_PS_TCP)) {
      HOLOSCAN_LOG_CRITICAL("Failed to create ID");
      return Status::CONNECT_FAILURE;
    }

    // Set up server address
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(server_port);
    addr.sin_addr.s_addr = server_addr;

    // Resolve the server's address
    if (rdma_resolve_addr(cm_id, nullptr, (struct sockaddr *)&addr, 2000)) {
      HOLOSCAN_LOG_CRITICAL("Failed to resolve address");
      return Status::CONNECT_FAILURE;
    }

    if (pd_map_.find(cm_id->verbs) == pd_map_.end()) {
      // PD not found, create it
      auto pd = ibv_alloc_pd(cm_id->verbs);
      if (pd == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate PD for device! {}", (void*)pd);
        return Status::CONNECT_FAILURE;
      }
      pd_map_[cm_id->verbs] = {};
      pd_map_[cm_id->verbs][cm_id->port_num] = pd;
    }

    // Wait for address resolution event
    if (rdma_get_cm_event(ec, &event)) {
      HOLOSCAN_LOG_CRITICAL("Failed to get CM event");
      return Status::CONNECT_FAILURE;
    }

    if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
      HOLOSCAN_LOG_CRITICAL("Unexpected event: {}", (int)event->event);
      rdma_ack_cm_event(event);
      return Status::CONNECT_FAILURE;
    }
    rdma_ack_cm_event(event);

    // Resolve route to server
    if (rdma_resolve_route(cm_id, 2000)) {
      HOLOSCAN_LOG_CRITICAL("Failed to resolve route");
      return Status::CONNECT_FAILURE;
    }

    // Wait for route resolution event
    if (rdma_get_cm_event(ec, &event)) {
      HOLOSCAN_LOG_CRITICAL("Failed to get CM event");
      return Status::CONNECT_FAILURE;
    }

    if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
      HOLOSCAN_LOG_CRITICAL("Unexpected event: {}", (int)event->event);
      rdma_ack_cm_event(event);
      return Status::CONNECT_FAILURE;
    }
    rdma_ack_cm_event(event);

    // Set up connection parameters
    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.responder_resources = 1;
    conn_param.initiator_depth = 1;
    conn_param.rnr_retry_count = 7;

    // Connect to server
    if (rdma_connect(cm_id, &conn_param)) {
      HOLOSCAN_LOG_CRITICAL("Failed to connect to server");
      return Status::CONNECT_FAILURE;
    }

    // Wait for connection established event
    if (rdma_get_cm_event(ec, &event)) {
      HOLOSCAN_LOG_CRITICAL("Failed to get CM event");
      return Status::CONNECT_FAILURE;
    }

    if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
      HOLOSCAN_LOG_CRITICAL("Unexpected event: {}", (int)event->event);
      rdma_ack_cm_event(event);
      return Status::CONNECT_FAILURE;
    }
    rdma_ack_cm_event(event);

    // Store the connection ID for later use
    mutex_.lock();
    rdma_thread_params client_params;
    client_params.client_id = cm_id;
    client_params.pd = pd_map_[cm_id->verbs][cm_id->port_num];
    client_params.if_idx = cm_id->port_num;
    client_params.queue_idx = client_q_params_.size();

    setup_thread_params(&client_params);
    worker_threads_[cm_id] = std::thread(&RdmaMgr::rdma_thread, this, false, std::move(client_params));
    client_q_params_[cm_id] = client_params;
    mutex_.unlock();

    *conn_id = reinterpret_cast<uintptr_t>(cm_id);
    HOLOSCAN_LOG_INFO("Successfully connected to server {} on port {}", 
        server_addr, server_port);

    rdma_destroy_event_channel(ec);

    return Status::SUCCESS;
  }


  void RdmaMgr::run() {
    int ret;

    HOLOSCAN_LOG_INFO("Starting RDMA CM main thread");

    if (set_affinity(cfg_.common_.master_core_) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to set master core affinity");
      return;
    }

    int num_ib_devices;    
    ibv_context **ib_devices = rdma_get_devices(&num_ib_devices);
    if (num_ib_devices == 0) {
      HOLOSCAN_LOG_CRITICAL("No RDMA-capable devices found!");
      return;
    }

    HOLOSCAN_LOG_INFO("Found {} RDMA-capable devices", num_ib_devices);

    // Create a channel for events. Only the master thread reads from this channel
    cm_event_channel_ = rdma_create_event_channel();    
    if (cm_event_channel_ == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to create a CM channel: {}", errno);
      return;
    }

    // Set the channel to non-blocking so we can check for SIGINT
    int flags = fcntl(cm_event_channel_->fd, F_GETFL);
    fcntl(cm_event_channel_->fd, F_SETFL, flags | O_NONBLOCK);


    // Start an RDMA server on each server interface
    for (const auto &intf: cfg_.ifs_) {
      if (intf.rdma_.mode_ != RDMAMode::SERVER) {
        continue;
      }

      // Initialize all the setup before the main loop
      struct sockaddr_in server_addr;
      memset(&server_addr, 0, sizeof(server_addr));
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(intf.rdma_.port_);  // Make sure port is set

      // Convert IP address string to network address
      if (inet_pton(AF_INET, intf.address_.c_str(), &server_addr.sin_addr) != 1) {
          HOLOSCAN_LOG_ERROR("Failed to convert IP address: {}", intf.address_);
          return;
      }

      HOLOSCAN_LOG_INFO("Successfully created CM event channel");

      struct rdma_cm_id *s_id;
      ret = rdma_create_id(cm_event_channel_, &s_id, nullptr, RDMA_PS_TCP);
      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to create RDMA server ID {}", errno);
        return;
      }

      pd_params_[s_id] = {};
      pd_params_[s_id].server_id = s_id;
      pd_params_[s_id].if_idx = intf.port_id_;

      HOLOSCAN_LOG_INFO("Created RDMA server on {}:{} successfully", 
        intf.address_, intf.rdma_.port_);

      ret = rdma_bind_addr(s_id, reinterpret_cast<struct sockaddr*>(&server_addr));
      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to bind for RDMA server: {} ({})", strerror(errno), errno);
        return;
      }

      ret = rdma_listen(s_id, intf.rx_.queues_.size());
      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to listen for RDMA server: {}", errno);
        return;
      }

      HOLOSCAN_LOG_INFO("RDMA server successfully started on {} queues", intf.rx_.queues_.size());
    }


    if (rdma_register_cfg_mrs() < 0) {
      HOLOSCAN_LOG_ERROR("Failed to register MRs");
      return;
    }

    HOLOSCAN_LOG_INFO("Entering CM main loop");

    // Our master thread's job is to wait on connections from clients, set up all the needed
    // information for them (QPs, MRs, etc), and spawn helper threads to monitor both TX and RX
    struct rdma_cm_event *cm_event = nullptr;    
    while (!rdma_force_quit.load()) {
      ret = rdma_get_cm_event(cm_event_channel_, &cm_event);
      if (ret != 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          // Sleep for 500 microseconds and try again  
          usleep(500);
          continue;
        }

        HOLOSCAN_LOG_INFO("Failed to get CM event: {}", errno);
        continue;
      }

      if (cm_event->status != 0) {
        HOLOSCAN_LOG_ERROR("Error status received in CM event: {}", cm_event->status);
        if (!rdma_ack_cm_event(cm_event)) {
          return;
        }

        continue;
      }

      switch (cm_event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST: {
          HOLOSCAN_LOG_INFO("Received new connection request for client ID {}", 
              (void*)cm_event->id);

          ack_event(cm_event);

          const auto listen_iter = pd_params_.find(cm_event->id);
          if (listen_iter == pd_params_.end()) {
            HOLOSCAN_LOG_ERROR("Failed to find server ID for {}", (void*)cm_event->id);
            break;
          }
          else {
            rdma_thread_params client_params;
            client_params.client_id = cm_event->id;
            if (pd_map_.find(cm_event->id->verbs) == pd_map_.end()) {
              // Create a new PD for this device
              HOLOSCAN_LOG_INFO("Creating new PD for device {} on port {}", 
                  (void*)cm_event->id->verbs, cm_event->id->port_num);
              auto pd = ibv_alloc_pd(cm_event->id->verbs);
              if (pd == nullptr) {
                HOLOSCAN_LOG_CRITICAL("Failed to allocate PD for device! {}", (void*)pd);
                break;
              }

              pd_map_[cm_event->id->verbs] = {};
              pd_map_[cm_event->id->verbs][cm_event->id->port_num] = pd;
            }
            
            client_params.pd = pd_map_[cm_event->id->verbs][cm_event->id->port_num];
            client_params.if_idx = cm_event->id->port_num;
            client_params.queue_idx = server_q_params_.size();

            if (server_q_params_.size() >= 
                cfg_.ifs_[listen_iter->second.if_idx].rx_.queues_.size()) {
              HOLOSCAN_LOG_CRITICAL("More RDMA connections than RX queues! Dropping connection");
              rdma_reject(client_params.client_id, nullptr, 0);
              break;
            }

            mutex_.lock();
            if (server_q_params_.find(cm_event->id) != 
                server_q_params_.end()) {
              HOLOSCAN_LOG_CRITICAL("Client ID {} already exists in server ID {}", 
                  (void*)cm_event->id, (void*)listen_iter->second.server_id);
              rdma_reject(client_params.client_id, nullptr, 0);
              mutex_.unlock();
              break;
            }

            server_q_params_[cm_event->id] = client_params;
            mutex_.unlock();

            struct rdma_conn_param conn_param = {};
            conn_param.responder_resources = 1;
            conn_param.initiator_depth = 1;
            conn_param.rnr_retry_count = 7;     

            if (rdma_accept(client_params.client_id , &conn_param) != 0) {
              HOLOSCAN_LOG_CRITICAL("Failed to accept connection: {}", errno);
              rdma_reject(client_params.client_id, nullptr, 0);
              continue;
            }
          }
          break;
        }
        case RDMA_CM_EVENT_ESTABLISHED: {
          bool found = false;
          mutex_.lock();
          const auto client_iter = server_q_params_.find(cm_event->id);

          // Client is found, set up QP and spawn a new thread
          if (client_iter != server_q_params_.end()) {
            setup_thread_params(&client_iter->second);
            HOLOSCAN_LOG_INFO("Configured queues for client. Launching thread");

            worker_threads_[cm_event->id] = std::thread(&RdmaMgr::rdma_thread, this, true, std::move(client_iter->second));
            found = true;
          }
          mutex_.unlock();

          if (!found) {
            HOLOSCAN_LOG_CRITICAL("Received established event for unknown client ID {}", 
                (void*)cm_event->id);
            break;
          }

          HOLOSCAN_LOG_INFO("Received established event for client ID {}", (void*)cm_event->id);
          break;
        }
        case RDMA_CM_EVENT_DISCONNECTED: {
          HOLOSCAN_LOG_INFO("Received disconnected event for client ID {}", (void*)cm_event->id);

          // Find client across all servers and remove it 
          mutex_.lock();
          const auto client_iter = server_q_params_.find(cm_event->id);
          if (client_iter != server_q_params_.end()) {
            worker_threads_[cm_event->id].join();
            server_q_params_.erase(client_iter);
            HOLOSCAN_LOG_INFO("Joined and removed client thread for ID {}", (void*)cm_event->id);
          }
          else {
            HOLOSCAN_LOG_CRITICAL("Received disconnected event for unknown client ID {}", 
                (void*)cm_event->id);
          }
          mutex_.unlock();
          break;
        }
        default: {
          HOLOSCAN_LOG_INFO("Cannot handle event type {}", (int)cm_event->event);
          break;
        }
      }     
    }

    // Join any client threads we had spawned
    HOLOSCAN_LOG_INFO("Waiting for server TX/RX threads to complete");
    for (auto &thread : worker_threads_) {
      thread.second.join();
    }

    HOLOSCAN_LOG_INFO("Finished cleaning up TX/RX workers");
  }

  int RdmaMgr::setup_pools_and_rings(int max_rx_batch, int max_tx_batch) {
    HOLOSCAN_LOG_DEBUG("Setting up RX ring");

    rx_ring =
        rte_ring_create("RX_RING", 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
    if (rx_ring == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
      return -1;
    }

    HOLOSCAN_LOG_DEBUG("Setting up RX meta pool");
    rx_meta = rte_mempool_create("RX_META_POOL",
                                (1U << 6) - 1U,
                                sizeof(BurstParams),
                                0,
                                0,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                rte_socket_id(),
                                0);
    if (rx_meta == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to allocate RX meta pool!");
      return -1;
    } 

    HOLOSCAN_LOG_DEBUG("Setting up TX meta pool");
    tx_meta = rte_mempool_create("TX_META_POOL",
                                (1U << 6) - 1U,
                                sizeof(BurstParams),
                                0,
                                0,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                rte_socket_id(),
                                0);
    if (tx_meta == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to allocate TX meta pool!");
      return -1;
    }

    for (const auto& intf : cfg_.ifs_) {
      for (const auto& q : intf.tx_.queues_) {
        const auto append =
            "P" + std::to_string(intf.port_id_) + "_Q" + std::to_string(q.common_.id_);

        auto name = "TX_RING_" + append;
        HOLOSCAN_LOG_INFO("Setting up TX ring {}", name);
        uint32_t key = (intf.port_id_ << 16) | q.common_.id_;
        tx_rings[key] = rte_ring_create(
            name.c_str(), 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
        if (tx_rings[key] == nullptr) {
          HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
          return -1;
        }
      }
    }

    return 0;          
  }

  void RdmaMgr::free_rx_burst(BurstParams* burst) {
    rte_mempool_put(rx_meta, burst);
  }

  void RdmaMgr::free_tx_burst(BurstParams* burst) {
    // const uint32_t key = (burst->rdma_hdr.port_id << 16) | burst->rdma_hdr.q_id;
    // const auto burst_pool = tx_burst_buffers.find(key);

    // for (int seg = 0; seg < burst->rdma_hdr.num_segs; seg++) {
    //   rte_mempool_put(burst_pool->second, (void*)burst->pkts[seg]);
    // }
  } 

  std::string RdmaMgr::generate_random_string(int len) {
    const char tokens[] = "abcdefghijklmnopqrstuvwxyz";
    std::string tmp;

    for (int i = 0; i < len; i++) { tmp += tokens[rand() % (sizeof(tokens) - 1)]; }

    return tmp;
  }   


  void RdmaMgr::initialize() {
    bool server = false;
    bool client = false;
    int ret;

    /* Initialize DPDK params */
    constexpr int max_nargs = 32;
    constexpr int max_arg_size = 64;
    char** _argv;
    _argv = (char**)malloc(sizeof(char*) * max_nargs);
    for (int i = 0; i < max_nargs; i++) { _argv[i] = (char*)malloc(max_arg_size); }

    int arg = 0;
    std::string cores = std::to_string(cfg_.common_.master_core_) + ",";  // Master core must be first
    std::set<std::string> ifs;

    // Get GPU PCIe BDFs since they're needed to pass to DPDK
    int if_num = 0;
    for (auto& intf : cfg_.ifs_) {
      ifs.emplace(intf.address_);
      for (const auto& q : intf.rx_.queues_) { cores += q.common_.cpu_core_ + ","; }
      for (const auto& q : intf.tx_.queues_) { cores += q.common_.cpu_core_ + ","; }
      intf.port_id_ = if_num++;
    }

    cores = cores.substr(0, cores.size() - 1);

    strncpy(_argv[arg++], "adv_net_operator", max_arg_size - 1);
    strncpy(_argv[arg++],
            (std::string("--file-prefix=") + generate_random_string(10)).c_str(),
            max_arg_size - 1);
    strncpy(_argv[arg++], "-l", max_arg_size - 1);
    strncpy(_argv[arg++], cores.c_str(), max_arg_size - 1);

    HOLOSCAN_LOG_INFO(
        "Setting DPDK log level to: {}",
        DpdkLogLevel::to_description_string(DpdkLogLevel::from_ano_log_level(cfg_.log_level_)));

    DpdkLogLevelCommandBuilder cmd(cfg_.log_level_);
    for (auto& c : cmd.get_cmd_flags_strings()) {
      strncpy(_argv[arg++], c.c_str(), max_arg_size - 1);
    }

    _argv[arg] = nullptr;
    std::string dpdk_args = "";
    for (int ac = 0; ac < arg; ac++) { dpdk_args += std::string(_argv[ac]) + " "; }

    HOLOSCAN_LOG_INFO("DPDK EAL arguments: {}", dpdk_args);

    ret = rte_eal_init(arg, _argv);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Invalid EAL arguments: {}", rte_errno);
      return;
    }    

    // Set up memory region sizes
    for (auto& mr : cfg_.mrs_) {
      mr.second.adj_size_ = mr.second.buf_size_;
    }    

    for (const auto& intf : cfg_.ifs_) {
      if (intf.rdma_.mode_ == RDMAMode::SERVER) {
        server = true;
      }
      else {
        client = true;
      }
    }

    main_thread_ = std::thread(&RdmaMgr::run, this);
  }

  void RdmaMgr::shutdown() {
    HOLOSCAN_LOG_INFO("RDMA ANO manager shutting down");
    rdma_force_quit.store(true);

    HOLOSCAN_LOG_INFO("Waiting for main thread to complete");
    main_thread_.join();    
  } 

  void RdmaMgr::init_client() {
    // TODO: Implement client initialization
    //return Status::SUCCESS;
  }


  Status RdmaMgr::get_tx_packet_burst(BurstParams *burst) {
    // TODO: Implement get_tx_packet_burst
    return Status::SUCCESS;
  }

  bool RdmaMgr::is_tx_burst_available(BurstParams *burst) {
    // TODO: Implement tx_burst_available
    return true;
  }

  std::optional<uint16_t> RdmaMgr::get_port_from_ifname(const std::string &name) {
    // Look up port owning the IP address passed in
    for (const auto& intf : cfg_.ifs_) {
      HOLOSCAN_LOG_INFO("intf.name_ {} {}", intf.name_, name);
      if (intf.name_ == name) {
        return intf.port_id_;
      }
    }
    
    return std::nullopt;
  }

  Status RdmaMgr::get_rx_burst(BurstParams **burst, int port, int q) {
    // TODO: Implement get_rx_burst
    return Status::SUCCESS;
  }

  Status RdmaMgr::set_packet_tx_time(BurstParams *burst, int idx, uint64_t timestamp) {
    // TODO: Implement set_pkt_tx_time
    return Status::SUCCESS;
  }

  void RdmaMgr::free_rx_metadata(BurstParams *burst) {
    // TODO: Implement free_rx_metadata
  }

  void RdmaMgr::free_tx_metadata(BurstParams *burst) {
    // TODO: Implement free_tx_metadata
  }

  Status RdmaMgr::get_tx_metadata_buffer(BurstParams **burst) {
    // TODO: Implement get_tx_meta_buf
    return Status::SUCCESS;
  }

  Status RdmaMgr::send_tx_burst(BurstParams *burst) {
    // TODO: Implement send_tx_burst
    return Status::SUCCESS;
  }

  void RdmaMgr::print_stats() {
    // TODO: Implement print_stats
  }

  // Also need destructor implementation
  RdmaMgr::~RdmaMgr() {
    // TODO: Implement cleanup logic
  }

  // RDMA-specific functions that were declared but not implemented
  Status RdmaMgr::register_mr(std::string name, int intf, void *addr, size_t len, int flags) {
    // TODO: Implement register_mr
    return Status::SUCCESS;
  }
}