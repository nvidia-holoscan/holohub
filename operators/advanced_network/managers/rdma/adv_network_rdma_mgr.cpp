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

namespace holoscan::ops {
  std::atomic<bool> rdma_force_quit = false;

  bool RdmaMgr::set_config_and_initialize(const AdvNetConfigYaml &cfg) {
    if (!this->initialized_) {
      cfg_ = cfg;
      initialize();
    }    

    return true;
  }

  // Common ANO functions
  AdvNetStatus RdmaMgr::set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                    const std::initializer_list<int>& lens) {
// populate later
    return AdvNetStatus::SUCCESS;
  }  

  void* RdmaMgr::get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
    return burst->pkts[seg][idx];
  }

  void* RdmaMgr::get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
    return burst->pkts[0][idx];
  }  
  
  uint16_t RdmaMgr::get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
    return burst->pkt_lens[seg][idx];
  }

  uint16_t RdmaMgr::get_pkt_len(AdvNetBurstParams* burst, int idx) {
    return burst->pkt_lens[0][idx];
  }

  AdvNetStatus RdmaMgr::set_eth_hdr(AdvNetBurstParams *burst, int idx,
                                    char *dst_addr) {
    HOLOSCAN_LOG_CRITICAL("Cannot set Ethernet header in RDMA mode");
    return AdvNetStatus::NOT_SUPPORTED;
  }

  AdvNetStatus RdmaMgr::set_ipv4_hdr(AdvNetBurstParams *burst, int idx,
                                    int ip_len,
                                    uint8_t proto,
                                    unsigned int src_host,
                                    unsigned int dst_host) {
    HOLOSCAN_LOG_CRITICAL("Cannot set IPv4 header in RDMA mode");
    return AdvNetStatus::NOT_SUPPORTED;
  }

  AdvNetStatus RdmaMgr::set_udp_hdr(AdvNetBurstParams *burst,
                                    int idx,
                                    int udp_len,
                                    uint16_t src_port,
                                    uint16_t dst_port) {
    HOLOSCAN_LOG_CRITICAL("Cannot set UDP header in RDMA mode");
    return AdvNetStatus::NOT_SUPPORTED;
  }

  AdvNetStatus RdmaMgr::set_udp_payload(AdvNetBurstParams *burst, int idx,
                                    void *data, int len) {
    HOLOSCAN_LOG_CRITICAL("Cannot set UDP payload in RDMA mode");
    return AdvNetStatus::NOT_SUPPORTED;
  }  

  uint64_t RdmaMgr::get_burst_tot_byte(AdvNetBurstParams* burst) {
    return 0;
  }

  AdvNetBurstParams* RdmaMgr::create_burst_params() {
    return new AdvNetBurstParams();
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

  int RdmaMgr::rdma_register_mr(const MemoryRegion &mr, void *ptr, int port_id) {
    rdma_mr_params params;
    params.params_ = mr;

    const auto pd = pd_map_.find(port_id);
    if (pd == pd_map_.end()) {
      HOLOSCAN_LOG_CRITICAL("Cannot find MR interface {} in PD mapping", port_id);
      return -1;
    }

    int access = mr_access_to_ibv(mr.access_);
    params.mr_ = ibv_reg_mr(pd->second, ptr, mr.buf_size_ * mr.num_bufs_, access);
    if (params.mr_ == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed to register MR {}", mr.name_);
      return -1;
    }

    HOLOSCAN_LOG_INFO("Successfully registered MR {} with {} bytes", mr.name_, mr.buf_size_ * mr.num_bufs_);

    mrs_[mr.name_] = params;  

    return 0;  
  }

  int RdmaMgr::rdma_register_cfg_mrs() {
    HOLOSCAN_LOG_INFO("Registering memory regions");

    if (allocate_memory_regions() != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to allocate memory");
      return -1;
    } 

    for (const auto &mr: cfg_.mrs_) {
      const auto port_id = get_intf_from_mr(mr.second.name_);
      if (port_id == -1) {
        HOLOSCAN_LOG_CRITICAL("Cannot find interface for MR {}", mr.second.name_);
        return -1;
      }

      int ret = rdma_register_mr(mr.second, ar_[mr.second.name_].ptr_, port_id);
      if (ret < 0) {
        return ret;
      }
    }

    // Register all the MRs for exchanging keys
    for (int p = 0; p < cfg_.ifs_.size(); p++) {
      const std::string name = std::string("lkey_") + std::to_string(p);
      MemoryRegion mr{
        name,
        cfg_.ifs_[p].name_,
        MemoryKind::HOST,
        MEM_ACCESS_LOCAL,
        sizeof(rdma_key_xchg),
        MAX_NUM_MR
      };

      lkey_mrs_.emplace_back(rdma_key_xchg{}); // Dummy entry where data will be written

      if (rdma_register_mr(mr, &lkey_mrs_[p], cfg_.ifs_[p].port_id_) < 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to register key exchange MR for interface {}", p);
        return -1;
      }
    }

    return 0;
  }

  AdvNetStatus RdmaMgr::wait_on_key_xchg() {
    return AdvNetStatus::SUCCESS;
  }

  /**
   * Set up all parameters needed for a newly-connected client
  */
  int RdmaMgr::setup_client_params_for_server(rdma_server_params *sparams, int if_idx) {
    // RX/TX queues should be symmetric with RDMA
    const auto num_queues = cfg_.ifs_[if_idx].rx_.queues_.size();
    for (int qi = 0; qi < num_queues; qi++) {
      rdma_qp_params qp_params;

      qp_params.rx_cq = ibv_create_cq(sparams->client_id->verbs, 
        MAX_CQ, nullptr, nullptr,	0);
      if (qp_params.rx_cq == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to create RX queue pair! {}", errno);
        return -1;
      }  

      qp_params.tx_cq = ibv_create_cq(sparams->client_id->verbs, 
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

      if (cfg_.ifs_[if_idx].rdma_.xmode_ == RDMATransportMode::RC) {
        qp_params.qp_attr.qp_type = IBV_QPT_RC;
      }
      else if (cfg_.ifs_[if_idx].rdma_.xmode_ == RDMATransportMode::UC) {
        qp_params.qp_attr.qp_type = IBV_QPT_UC;
      }
      else {
        HOLOSCAN_LOG_ERROR("RDMA transport mode {} not supported!", 
              static_cast<int>(cfg_.ifs_[if_idx].rdma_.xmode_));
        return -1;
      }

      // Share the CQ between TX and RX
      qp_params.qp_attr.recv_cq = qp_params.rx_cq;
      qp_params.qp_attr.send_cq = qp_params.tx_cq;

      int ret = rdma_create_qp(sparams->client_id, sparams->pd, &qp_params.qp_attr);
      if (ret != 0) {
	      HOLOSCAN_LOG_CRITICAL("Failed to create QP: {}", errno);
	      return -1;
      } 

      // Create POSIX message queues for talking to client
      struct mq_attr attr;

      attr.mq_flags = 0;
      attr.mq_maxmsg = 128;
      attr.mq_msgsize = sizeof(AdvNetBurstParams);
      attr.mq_curmsgs = 0;

      const int q = 0; // Add multiple queues later
      std::string q_name = "I" + std::to_string(cfg_.ifs_[if_idx].port_id_) + 
                           "_Q" + std::to_string(q);
      qp_params.rx_mq = mq_open((q_name + "_RX").c_str(), O_CREAT | O_WRONLY, 0644, &attr);
      qp_params.tx_mq = mq_open((q_name + "_TX").c_str(), O_CREAT | O_WRONLY, 0644, &attr);

      if (qp_params.rx_mq == (mqd_t)-1 || 
          qp_params.tx_mq == (mqd_t)-1) {
        HOLOSCAN_LOG_CRITICAL("Failed to create message queues for {}", q_name);
        return -1;
      }

      sparams->qp_params.emplace_back(qp_params);        
    }

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
   * Worker thread for a server SQ
  */
  void RdmaMgr::server_tx(int if_idx, int q) {
    struct ibv_wc wc;
    int num_comp;
    const auto &qref = cfg_.ifs_[if_idx].tx_.queues_[q];
    const auto &rdma_qref = sparams_[if_idx].qp_params[q];
    const long cpu_core = strtol(qref.common_.cpu_core_.c_str(), NULL, 10);

    if (set_affinity(cpu_core) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to set TX core affinity");
      return;
    }

    HOLOSCAN_LOG_INFO("Affined TX thread to core {}", cpu_core);

    // Main TX loop. Wait for send requests from the transmitters to arrive for sending. Also
    // periodically poll the CQ.
    while (!rdma_force_quit.load()) {
      while ((num_comp = ibv_poll_cq(rdma_qref.tx_cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", wc.status, wc.wr_id, wc.opcode);
          continue;
        }

        if (wc.opcode == IBV_WC_RDMA_READ) {
          // AdvNetBurstParams read_burst;
          // read_burst.hdr.hdr.

          // out_wr_[cur_wc_id_].wr_id = cur_wc_id_;
          // out_wr_[cur_wc_id_].done  = false;
          // out_wr_[cur_wc_id_].mr = *(endpt->second);
          // out_wr_[cur_wc_id_].mr.ptr = burst.cpu_pkts[p];
          // cur_wc_id_++;

          // auto ret = mq_send(rdma_qref.rx_mq, &read_burst, sizeof(read_burst), 10);
          // if (ret != 0) {
          //   HOLOSCAN_LOG_CRITICAL("Failed to send to message queue: {}", errno);
          //   continue;
          // }
        }
      }

      // Now handle any incoming sends
      AdvNetBurstParams burst;
      ssize_t bytes = mq_receive(rdma_qref.tx_mq, reinterpret_cast<char*>(&burst), sizeof(burst), nullptr);
      if (bytes > 0) {
        const auto endpt = endpoints_.find(reinterpret_cast<struct rdma_cm_id*>(burst.hdr.hdr.rdma.dst_key));
        if (endpt == endpoints_.end()) {
          HOLOSCAN_LOG_ERROR("Trying to send to client {}, but that client is not connected", burst.hdr.hdr.rdma.dst_key);
          continue;
        }

        const auto local_mr = mrs_.find(burst.hdr.hdr.rdma.local_mr_name);
        if (local_mr == mrs_.end()) {
          HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry", burst.hdr.hdr.rdma.local_mr_name);
          free_tx_burst(&burst);
          continue;
        }

        switch (burst.hdr.hdr.opcode) {
          case AdvNetOpCode::SEND:
          { // perform send operation
            // Currently we expect SEND operations to be rare, and certainly not with thousands of
            // packets. For that reason we post one at a time and do not try to batch.
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
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
          case AdvNetOpCode::RDMA_WRITE: [[fall_through]];
          case AdvNetOpCode::RDMA_WRITE_IMM:
          {
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
              ibv_send_wr wr;
              ibv_send_wr *bad_wr;
              ibv_sge sge;

              memset(&wr, 0, sizeof(wr));
              sge.addr      = (uint64_t)burst.pkts[0][p];
              sge.length    = (uint32_t)burst.pkt_lens[0][p];
              sge.lkey      = local_mr->second.mr_->lkey;
              wr.sg_list    = &sge;
              wr.num_sge    = 1;
              if (burst.hdr.hdr.opcode==AdvNetOpCode::RDMA_WRITE) {
                wr.opcode   = IBV_WR_RDMA_WRITE;
              }
              else {
                wr.opcode   = IBV_WR_RDMA_WRITE_WITH_IMM;
                wr.imm_data = htonl(burst.hdr.hdr.rdma.imm);
              }
 
              wr.send_flags = IBV_SEND_SIGNALED;

              // Look up remote key
              const auto remote_mr = endpt->second.find(burst.hdr.hdr.rdma.remote_mr_name);
              if (remote_mr == endpt->second.end()) {
                HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry for client {}", 
                      burst.hdr.hdr.rdma.remote_mr_name, burst.hdr.hdr.rdma.dst_key);
                free_tx_burst(&burst);
                continue;
              }              
              wr.wr.rdma.rkey = remote_mr->second.key;
              wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(burst.hdr.hdr.rdma.raddr);

              int ret = ibv_post_send(endpt->first->qp, &wr, &bad_wr);
              if (ret != 0) {
                HOLOSCAN_LOG_CRITICAL("Failed to post SEND request, errno: {}", errno);
                free_tx_burst(&burst);
                continue;
              }
            }
            break;
          }
          case AdvNetOpCode::RDMA_READ:
          {
            for (int p = 0; p < burst.hdr.hdr.num_pkts; p++) {
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
              const auto remote_mr = endpt->second.find(burst.hdr.hdr.rdma.remote_mr_name);
              if (remote_mr == endpt->second.end()) {
                HOLOSCAN_LOG_CRITICAL("Couldn't find MR with name {} in registry for client {}", 
                      burst.hdr.hdr.rdma.remote_mr_name, burst.hdr.hdr.rdma.dst_key);
                free_tx_burst(&burst);
                continue;
              }              
              wr.wr.rdma.rkey  = remote_mr->second.key;
              wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(burst.hdr.hdr.rdma.raddr);

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
  }

  /**
   * Worker thread for a server RQ
  */
  void RdmaMgr::server_rx(int if_idx, int q) {
    struct ibv_wc wc;
    int num_comp;
    const auto &qref = cfg_.ifs_[if_idx].tx_.queues_[q];
    const auto &rdma_qref = sparams_[if_idx].qp_params[q];
    const long cpu_core = strtol(qref.common_.cpu_core_.c_str(), NULL, 10);


    if (set_affinity(cpu_core) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to set RX core affinity");
      return;
    }

    HOLOSCAN_LOG_INFO("Affined RX thread to core {}", cpu_core);

    // Main TX loop. Wait for send requests from the transmitters to arrive for sending. Also
    // periodically poll the CQ.
    while (!rdma_force_quit.load()) {
      while ((num_comp = ibv_poll_cq(rdma_qref.rx_cq, 1, &wc)) != 0) {
        if (wc.status != IBV_WC_SUCCESS) {
          HOLOSCAN_LOG_ERROR("CQ error {} for WRID {} and opcode {}", wc.status, wc.wr_id, wc.opcode);
        }
      }    
    }
  }

  void RdmaMgr::run_server() {
    int ret;

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

    // Start an RDMA server on each interface specified
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
      cm_server_id_.push_back(s_id);

      HOLOSCAN_LOG_INFO("Created RDMA server ID successfully");

      ret = rdma_bind_addr(s_id, reinterpret_cast<struct sockaddr*>(&server_addr));
      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to bind for RDMA server: {} ({})", strerror(errno), errno);
        return;
      }

      // Create a protection domain
      auto pd = ibv_alloc_pd(s_id->verbs);
      if (pd == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate PD for device! {}", (void*)pd);
        return;
      }

      pd_map_[intf.port_id_] = pd;

      ret = rdma_listen(s_id, MAX_RDMA_CONNECTIONS);
      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to listen for RDMA server: {}", errno);
        return;
      }

      HOLOSCAN_LOG_INFO("RDMA server successfully started");
    }


    if (rdma_register_cfg_mrs() < 0) {
      HOLOSCAN_LOG_ERROR("Failed to register MRs");
      return;
    }

    HOLOSCAN_LOG_INFO("Entering server main loop");

    // Create enough server parameters for each interface
    sparams_.resize(10); // Temporary -- fix later

    // Our master thread's job is to wait on connections from clients, set up all the needed
    // information for them (QPs, MRs, etc), and spawn client threads to monitor both TX and RX
    struct rdma_cm_event *cm_event = nullptr;    
    while (true) {
      ret = rdma_get_cm_event(cm_event_channel_, &cm_event);
      if (ret != 0) {
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
          rdma_server_params sparams{};
          int listen_idx;
          sparams.client_id = cm_event->id;
          const auto listen_id = cm_event->listen_id;          
          ack_event(cm_event);

          const auto listen_iter = std::find(cm_server_id_.begin(), cm_server_id_.end(), listen_id);
          if (listen_iter == cm_server_id_.end()) {
            HOLOSCAN_LOG_ERROR("Failed to find listener ID for {}", (void*)listen_id);
            break;
          }
          else {
            // resize sparams_ for number of interfaces and add to array. finish posix mq work
            listen_idx = listen_iter - cm_server_id_.begin();
            setup_client_params_for_server(&sparams, listen_idx);
            HOLOSCAN_LOG_INFO("Configured queues for client. Launching threads");

            sparams_[listen_idx] = sparams;

            // Spawn a new TX and RX thread for each QP
            const auto num_queues = 1; //cfg_.rx[listen_idx].queues_.size();
            for (int q = 0; q < num_queues; q++) {
              txrx_workers.emplace_back(std::thread(&RdmaMgr::server_tx, this, listen_idx, q));
              txrx_workers.emplace_back(std::thread(&RdmaMgr::server_rx, this, listen_idx, q));
            }
          }
          break;
        }
        default:
          HOLOSCAN_LOG_INFO("Cannot handle event type {}", cm_event->event);
      }     
    }

    // Join any client threads we had spawned
    HOLOSCAN_LOG_INFO("Waiting for server TX/RX threads to complete");
    for (auto &w: txrx_workers) {
      w.join();
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
                                sizeof(AdvNetBurstParams),
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
                                sizeof(AdvNetBurstParams),
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

  void RdmaMgr::free_rx_burst(AdvNetBurstParams* burst) {
    rte_mempool_put(rx_meta, burst);
  }

  void RdmaMgr::free_tx_burst(AdvNetBurstParams* burst) {
    // const uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
    // const auto burst_pool = tx_burst_buffers.find(key);

    // for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
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

    if (server) {
      std::thread t(&RdmaMgr::run_server, this);
      t.join();
    }
    
    if (client) {
      init_client();
      std::thread t(&RdmaMgr::run_client, this);
      t.join();
    }
  }

  void RdmaMgr::shutdown() {
    HOLOSCAN_LOG_INFO("RDMA ANO manager shutting down");
    rdma_force_quit.store(true);
  } 

  void RdmaMgr::init_client() {
    // TODO: Implement client initialization
    //return AdvNetStatus::SUCCESS;
  }

  void RdmaMgr::run_client() {
    // TODO: Implement client run logic
  }

  void RdmaMgr::run() {
    // TODO: Implement run logic
  }

  AdvNetStatus RdmaMgr::get_tx_pkt_burst(AdvNetBurstParams *burst) {
    // TODO: Implement get_tx_pkt_burst
    return AdvNetStatus::SUCCESS;
  }

  bool RdmaMgr::tx_burst_available(AdvNetBurstParams *burst) {
    // TODO: Implement tx_burst_available
    return true;
  }

  std::optional<uint16_t> RdmaMgr::get_port_from_ifname(const std::string &name) {
    // TODO: Implement get_port_from_ifname
    return std::nullopt;
  }

  AdvNetStatus RdmaMgr::get_rx_burst(AdvNetBurstParams **burst) {
    // TODO: Implement get_rx_burst
    return AdvNetStatus::SUCCESS;
  }

  AdvNetStatus RdmaMgr::set_pkt_tx_time(AdvNetBurstParams *burst, int idx, uint64_t timestamp) {
    // TODO: Implement set_pkt_tx_time
    return AdvNetStatus::SUCCESS;
  }

  void RdmaMgr::free_rx_meta(AdvNetBurstParams *burst) {
    // TODO: Implement free_rx_meta
  }

  void RdmaMgr::free_tx_meta(AdvNetBurstParams *burst) {
    // TODO: Implement free_tx_meta
  }

  AdvNetStatus RdmaMgr::get_tx_meta_buf(AdvNetBurstParams **burst) {
    // TODO: Implement get_tx_meta_buf
    return AdvNetStatus::SUCCESS;
  }

  AdvNetStatus RdmaMgr::send_tx_burst(AdvNetBurstParams *burst) {
    // TODO: Implement send_tx_burst
    return AdvNetStatus::SUCCESS;
  }

  void RdmaMgr::print_stats() {
    // TODO: Implement print_stats
  }

  // Also need destructor implementation
  RdmaMgr::~RdmaMgr() {
    // TODO: Implement cleanup logic
  }

  // RDMA-specific functions that were declared but not implemented
  AdvNetStatus RdmaMgr::register_mr(std::string name, int intf, void *addr, size_t len, int flags) {
    // TODO: Implement register_mr
    return AdvNetStatus::SUCCESS;
  }
}