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
#include <cuda.h>
#include <complex>
#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <sys/time.h>
#include "adv_network_dpdk_mgr.h"
#include "holoscan/holoscan.hpp"

using namespace std::chrono;

namespace holoscan::ops {

std::atomic<bool> force_quit = false;

struct TxWorkerParams {
  int port;
  int queue;
  uint32_t batch_size;
  struct rte_ring* ring;
  struct rte_mempool* meta_pool;
  struct rte_mempool* burst_pool;
  struct rte_ether_addr mac_addr;
};

struct RxWorkerParams {
  int port;
  int queue;
  int num_segs;
  uint64_t timeout_us;
  uint32_t batch_size;
  struct rte_ring* ring;
  struct rte_mempool* flowid_pool;
  struct rte_mempool* burst_pool;
  struct rte_mempool* meta_pool;
  uint64_t rx_pkts = 0;
};

/**
 * @brief Generic UDP packet structure
 *
 */
struct UDPPkt {
  struct rte_ether_hdr eth;
  struct rte_ipv4_hdr ip;
  struct rte_udp_hdr udp;
  uint8_t payload[];
} __attribute__((packed));

struct ExtraRxPacketInfo {
  uint16_t flow_id;
};

/**
 * A map of log level to a tuple of the description and command strings.
 */
const std::unordered_map<DpdkLogLevel::Level, std::tuple<std::string, std::string>>
    DpdkLogLevel::level_to_cmd_map = {{OFF, {"Disabled", "9"}},
                                      {EMERGENCY, {"Emergency", "emergency"}},
                                      {ALERT, {"Alert", "alert"}},
                                      {CRITICAL, {"Critical", "critical"}},
                                      {ERROR, {"Error", "error"}},
                                      {WARN, {"Warning", "warning"}},
                                      {NOTICE, {"Notice", "notice"}},
                                      {INFO, {"Info", "info"}},
                                      {DEBUG, {"Debug", "debug"}}};

const std::unordered_map<AnoLogLevel::Level, DpdkLogLevel::Level>
    DpdkLogLevel::ano_to_dpdk_log_level_map = {
        {AnoLogLevel::TRACE, DEBUG},
        {AnoLogLevel::DEBUG, DEBUG},
        {AnoLogLevel::INFO, INFO},
        {AnoLogLevel::WARN, WARN},
        {AnoLogLevel::ERROR, ERROR},
        {AnoLogLevel::CRITICAL, CRITICAL},
        {AnoLogLevel::OFF, OFF},
};

////////////////////////////////////////////////////////////////////////////////
///
///  \brief Init
///
////////////////////////////////////////////////////////////////////////////////
bool DpdkMgr::set_config_and_initialize(const AdvNetConfigYaml& cfg) {
  num_init++;

  if (!this->initialized_) {
    cfg_ = cfg;
    cpu_set_t mask;
    long nproc, i;

    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&DpdkMgr::initialize, this);
    t.join();

    this->initialized_ = true;
    if (!validate_config()) {
      HOLOSCAN_LOG_CRITICAL("Config validation failed");
      return false;
    }

    run();
  }

  return true;
}

int DpdkMgr::address_to_port(const std::string& addr) {
  for (const auto& intf : cfg_.ifs_) {
    if (intf.address_ == addr) { return intf.port_id_; }
  }

  return -1;
}

AdvNetStatus DpdkMgr::get_mac(int port, char* mac) {
  if (port > mac_addrs.size()) {
    HOLOSCAN_LOG_CRITICAL("Port {} out of range in get_mac() lookup");
    return AdvNetStatus::INVALID_PARAMETER;
  }

  memcpy(mac, reinterpret_cast<char*>(&mac_addrs[port]), sizeof(mac_addrs[port]));
  return AdvNetStatus::SUCCESS;
}

void DpdkMgr::adjust_memory_regions() {
  for (auto& mr : cfg_.mrs_) {
    // mr.second.buf_size_ = ((target_el_size + 3) / 4) * 4;
    mr.second.adj_size_ = mr.second.buf_size_ + RTE_PKTMBUF_HEADROOM;
    HOLOSCAN_LOG_INFO("Adjusting buffer size to {} for headroom", mr.second.adj_size_);
  }
}

AdvNetStatus DpdkMgr::map_mrs() {
  // Map every MR to every device for now
  for (const auto& intf : cfg_.ifs_) {
    struct rte_eth_dev_info dev_info;
    int ret = rte_eth_dev_info_get(intf.port_id_, &dev_info);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", intf.port_id_);
      return AdvNetStatus::NULL_PTR;
    }

    for (const auto& ext_mem_el : ext_pktmbufs_) {
      const auto& ext_mem = ext_mem_el.second;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      ret = rte_dev_dma_map(dev_info.device, ext_mem->buf_ptr, ext_mem->buf_iova, ext_mem->buf_len);
#pragma GCC diagnostic pop

      if (ret) {
        HOLOSCAN_LOG_CRITICAL(
            "Could not DMA map EXT memory: {} err={}", ret, rte_strerror(rte_errno));
        return AdvNetStatus::NULL_PTR;
      }

      HOLOSCAN_LOG_INFO(
          "Mapped external memory descriptor for {} to device {}", ext_mem->buf_ptr, intf.port_id_);
    }
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::register_mrs() {
  for (const auto& ar : ar_) {
    auto ext_mem = std::make_shared<struct rte_pktmbuf_extmem>();
    const auto& mr = cfg_.mrs_[ar.second.mr_name_];

    if (mr.kind_ == MemoryKind::HUGE) { continue; }

    ext_mem->buf_len = mr.ttl_size_;
    ext_mem->buf_iova = RTE_BAD_IOVA;
    ext_mem->buf_ptr = ar.second.ptr_;
    ext_mem->elt_size = mr.adj_size_;

    // GPUs have the largest page size vs CPUs, so just use that
    int ret = rte_extmem_register(
        ext_mem->buf_ptr, ext_mem->buf_len, NULL, ext_mem->buf_iova, GPU_PAGE_SIZE);
    if (ret) {
      HOLOSCAN_LOG_CRITICAL("Unable to register addr {}, ret {} errno {}",
                            ext_mem->buf_ptr,
                            ret,
                            rte_strerror(rte_errno));
      return AdvNetStatus::NULL_PTR;
    } else {
      HOLOSCAN_LOG_INFO("Successfully registered external memory for {}", mr.name_);
    }

    ext_pktmbufs_[mr.name_] = ext_mem;
  }

  return AdvNetStatus::SUCCESS;
}

void DpdkMgr::setup_accurate_send_scheduling_mask() {
  static bool done = false;
  if (done) { return; }

  static const rte_mbuf_dynfield dynfield_desc = {
      RTE_MBUF_DYNFIELD_TIMESTAMP_NAME,
      sizeof(uint64_t),
      .align = __alignof__(uint64_t),
  };

  static const rte_mbuf_dynflag dynflag_desc = {
      RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME,
  };

  timestamp_offset_ = rte_mbuf_dynfield_register(&dynfield_desc);
  if (timestamp_offset_ < 0) {
    HOLOSCAN_LOG_CRITICAL(
        "{} registration error: {}", RTE_MBUF_DYNFIELD_TIMESTAMP_NAME, rte_strerror(rte_errno));
    return;
  }

  int32_t dynflag_bitnum = rte_mbuf_dynflag_register(&dynflag_desc);
  if (dynflag_bitnum == -1) {
    HOLOSCAN_LOG_CRITICAL(
        "{} registration error: {}", RTE_MBUF_DYNFLAG_TX_TIMESTAMP_NAME, rte_strerror(rte_errno));
    return;
  }

  auto dynflag_shift = static_cast<uint8_t>(dynflag_bitnum);
  timestamp_mask_ = 1ULL << dynflag_shift;
  HOLOSCAN_LOG_INFO("Done setting up accurate send scheduling with mask {:x}", timestamp_mask_);
  done = true;
}

int DpdkMgr::numa_from_mem(const MemoryRegion& mr) {
  if (mr.kind_ == MemoryKind::DEVICE) {
    int val;
    if (cudaDeviceGetAttribute(&val, cudaDevAttrHostNumaId, mr.affinity_) != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to get NUMA node from device {}", mr.affinity_);
      return -1;
    }

    return val;
  } else {
    return mr.affinity_;
  }
}

std::string DpdkMgr::generate_random_string(int len) {
  const char tokens[] = "abcdefghijklmnopqrstuvwxyz";
  std::string tmp;

  for (int i = 0; i < len; i++) { tmp += tokens[rand() % (sizeof(tokens) - 1)]; }

  return tmp;
}

// HWS doesn't allow zero queues on an interface, so we make some dummy interfaces here for
// users that are only doing TX
void DpdkMgr::create_dummy_rx_q() {
  for (auto& intf : cfg_.ifs_) {
    auto& rx = intf.rx_;

    // With hardware steering we need to create a fake queue or DPDK will segfault when setting
    // the template parameters
    if (rx.queues_.size() == 0) {
      HOLOSCAN_LOG_INFO("Port {} has no RX queues. Creating dummy queue.", intf.port_id_);
      const std::string mr_name = "MR_Unused_P" + std::to_string(intf.port_id_);
      RxQueueConfig tmp_q;
      tmp_q.output_port_ = "none";
      tmp_q.common_.name_ = "UNUSED_P" + std::to_string(intf.port_id_) + "_Q0";
      tmp_q.common_.id_ = 0;
      tmp_q.common_.batch_size_ = 1;
      tmp_q.common_.split_boundary_ = 0;
      tmp_q.common_.cpu_core_ = "0";
      tmp_q.common_.mrs_.push_back(mr_name);
      tmp_q.common_.extra_queue_config_ = nullptr;
      rx.queues_.push_back(tmp_q);


      // Create unused MR
      MemoryRegion tmp_mr;
      tmp_mr.name_ = mr_name;
      tmp_mr.kind_ = MemoryKind::HUGE;
      tmp_mr.affinity_ = 0;
      tmp_mr.access_ = 0;
      tmp_mr.buf_size_ = JUMBOFRAME_SIZE;
      tmp_mr.num_bufs_ = 32768;
      tmp_mr.owned_ = true;
      cfg_.mrs_[mr_name] = tmp_mr;
    }
  }
}

void DpdkMgr::initialize() {
  int ret;
  uint16_t portid;

  static struct rte_eth_conf conf_eth_port = {
      .rxmode = {
              .mq_mode = RTE_ETH_MQ_RX_RSS,
              .offloads = 0,
          },
      .txmode = {
              .mq_mode = RTE_ETH_MQ_TX_NONE,
              .offloads = 0,
          },
      .rx_adv_conf = {
              .rss_conf = {.rss_key = NULL, .rss_hf = RTE_ETH_RSS_IP},
          },
  };

  for (auto& conf : local_port_conf) { conf = conf_eth_port; }

  /* Initialize DPDK params */
  constexpr int max_nargs = 32;
  constexpr int max_arg_size = 64;
  char** _argv;
  _argv = (char**)malloc(sizeof(char*) * max_nargs);
  for (int i = 0; i < max_nargs; i++) { _argv[i] = (char*)malloc(max_arg_size); }

  int arg = 0;
  std::string cores = std::to_string(cfg_.common_.master_core_) + ",";  // Master core must be first
  std::set<std::string> ifs;
  std::unordered_map<uint16_t, std::pair<uint16_t, uint16_t>> port_q_num;
  std::unordered_map<uint16_t, std::string> port_id_to_name;

  // Get GPU PCIe BDFs since they're needed to pass to DPDK
  for (const auto& intf : cfg_.ifs_) {
    ifs.emplace(intf.address_);
    for (const auto& q : intf.rx_.queues_) { cores += q.common_.cpu_core_ + ","; }

    for (const auto& q : intf.tx_.queues_) { cores += q.common_.cpu_core_ + ","; }
  }

  cores = cores.substr(0, cores.size() - 1);
  // Get a unique set of interfaces
  num_ports = ifs.size();
  HOLOSCAN_LOG_INFO("Attempting to use {} ports for high-speed network", num_ports);

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

  for (const auto& name : ifs) {
    strncpy(_argv[arg++], "-a", max_arg_size - 1);
    strncpy(_argv[arg++],
            (name + std::string(",txq_inline_max=0,dv_flow_en=2")).c_str(),
            max_arg_size - 1);
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

  for (int i = 0; i < num_ports; i++) {
    rte_eth_macaddr_get(i, &mac_addrs[i]);
  }

  create_dummy_rx_q();

  // Adjust the sizes to accommodate any padding/alignment restrictions by this library
  adjust_memory_regions();

  if (allocate_memory_regions() != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate memory");
    return;
  }

  if (register_mrs() != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to register MRs");
    return;
  }

  if (map_mrs() != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to map MRs");
    return;
  }

  // Build name to id mapping
  int max_rx_batch_size = 0;
  int max_tx_batch_size = 0;
  for (auto& intf : cfg_.ifs_) {
    ret = rte_eth_dev_get_port_by_name(intf.address_.c_str(), &intf.port_id_);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port number for {}", intf.name_.c_str());
      return;
    }

    struct rte_eth_dev_info dev_info;
    int ret = rte_eth_dev_info_get(intf.port_id_, &dev_info);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", intf.port_id_);
      return;
    }

    port_q_num[intf.port_id_] = {intf.rx_.queues_.size(), intf.tx_.queues_.size()};
    port_id_to_name[intf.port_id_] = intf.address_;

    HOLOSCAN_LOG_INFO("DPDK init ({}) -- RX: {} TX: {}",
                      intf.address_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");

    // Queue setup
    size_t max_pkt_size = 0;
    const auto& rx = intf.rx_;

    for (auto& q : rx.queues_) {
      HOLOSCAN_LOG_INFO("Configuring RX queue: {} ({}) on port {}",
                        q.common_.name_,
                        q.common_.id_,
                        intf.port_id_);
      auto q_backend = new DPDKQueueConfig{};
      max_rx_batch_size = std::max(max_rx_batch_size, q.common_.batch_size_);

      size_t q_packet_size = 0;
      for (int mr_num = 0; mr_num < q.common_.mrs_.size(); mr_num++) {
        std::string append = "_P" + std::to_string(intf.port_id_) + "_Q" +
                            std::to_string(q.common_.id_) + "_MR" + std::to_string(mr_num);
        std::string pool_name = std::string("RXP") + append;
        const auto& mr = cfg_.mrs_[q.common_.mrs_[mr_num]];

        if (mr.num_bufs_ < default_num_rx_desc) {
          HOLOSCAN_LOG_CRITICAL("Must have at least {} buffers in each RX MR", default_num_rx_desc);
          return;
        }

        struct rte_mempool* pool;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (mr.kind_ == MemoryKind::HUGE) {
          pool = rte_pktmbuf_pool_create(
              pool_name.c_str(), mr.num_bufs_, 0, 0, mr.adj_size_, numa_from_mem(mr));
        } else {
          auto pktmbuf = ext_pktmbufs_[q.common_.mrs_[mr_num]];
          pool = rte_pktmbuf_pool_create_extbuf(pool_name.c_str(),
                                                mr.num_bufs_,
                                                0,
                                                0,
                                                mr.adj_size_,
                                                numa_from_mem(mr),
                                                pktmbuf.get(),
                                                1);
        }
#pragma GCC diagnostic pop

        if (pool == nullptr) {
          HOLOSCAN_LOG_CRITICAL(
                "Could not create external memory mempool {}: mbufs={} elsize={} ptr={}",
                pool_name,
                mr.num_bufs_,
                mr.adj_size_,
                (void*)pool);
          return;
        }

        q_backend->pools.push_back(pool);
        HOLOSCAN_LOG_INFO("Created mempool {} : mbufs={} elsize={} ptr={}",
                          pool_name,
                          mr.num_bufs_,
                          mr.adj_size_,
                          (void*)pool);

        q_packet_size += mr.buf_size_;
      }

      max_pkt_size = std::max(max_pkt_size, q_packet_size);
      HOLOSCAN_LOG_INFO("Max packet size needed for RX: {}", max_pkt_size);

      // Multiple segments
      if (q.common_.mrs_.size() > 1) {
        memcpy(
            &q_backend->rxconf_qsplit, &dev_info.default_rxconf, sizeof(q_backend->rxconf_qsplit));

        q_backend->rxconf_qsplit.offloads =
            RTE_ETH_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
        q_backend->rxconf_qsplit.rx_nseg = q.common_.mrs_.size();

        q_backend->rx_useg.resize(q.common_.mrs_.size());
        q_backend->rxconf_qsplit.rx_seg = &q_backend->rx_useg[0];

        for (int seg = 0; seg < q.common_.mrs_.size(); seg++) {
          struct rte_eth_rxseg_split* rx_seg;

          rx_seg = &q_backend->rx_useg[seg].split;
          rx_seg->mp = q_backend->pools[seg];
          rx_seg->length = (seg == (q.common_.mrs_.size() - 1))
                              ? 0
                              : cfg_.mrs_[q.common_.mrs_[seg]].adj_size_ - RTE_PKTMBUF_HEADROOM;
          rx_seg->offset = 0;
        }

        local_port_conf[intf.port_id_].rxmode.offloads |=
            RTE_ETH_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
      }

      uint32_t key = (intf.port_id_ << 16) | q.common_.id_;
      rx_q_map_[key] = q_backend;
    }

    local_port_conf[intf.port_id_].rxmode.offloads |= RTE_ETH_RX_OFFLOAD_CHECKSUM;

    // TX now
    // For now make a single queue. Support more sophisticated TX on next release
    const auto& tx = intf.tx_;

    for (auto& q : tx.queues_) {
      HOLOSCAN_LOG_INFO("Configuring TX queue: {} ({}) on port {}",
                        q.common_.name_,
                        q.common_.id_,
                        intf.port_id_);
      auto q_backend = new DPDKQueueConfig{};
      max_tx_batch_size = std::max(max_tx_batch_size, q.common_.batch_size_);
      size_t q_packet_size = 0;

      for (int mr_num = 0; mr_num < q.common_.mrs_.size(); mr_num++) {
        std::string append = "_P" + std::to_string(intf.port_id_) + "_Q" +
                             std::to_string(q.common_.id_) + "_MR" + std::to_string(mr_num);
        std::string pool_name = std::string("TXP") + append;
        const auto& mr = cfg_.mrs_[q.common_.mrs_[mr_num]];

        if (mr.num_bufs_ < default_num_tx_desc) {
          HOLOSCAN_LOG_CRITICAL("Must have at least {} buffers in each TX MR", default_num_tx_desc);
          return;
        }

        struct rte_mempool* pool;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (mr.kind_ == MemoryKind::HUGE) {
          pool = rte_pktmbuf_pool_create(
              pool_name.c_str(), mr.num_bufs_, 0, 0, mr.buf_size_, numa_from_mem(mr));
        } else {
          auto pktmbuf = ext_pktmbufs_[q.common_.mrs_[mr_num]];
          pool = rte_pktmbuf_pool_create_extbuf(pool_name.c_str(),
                                                mr.num_bufs_,
                                                0,
                                                0,
                                                mr.buf_size_,
                                                numa_from_mem(mr),
                                                pktmbuf.get(),
                                                1);
        }
#pragma GCC diagnostic pop
        if (pool == nullptr) {
          HOLOSCAN_LOG_CRITICAL("Could not create external memory mempool");
          return;
        }

        q_backend->pools.push_back(pool);
        HOLOSCAN_LOG_INFO("Created mempool {} : mbufs={} elsize={} ptr={}",
                          pool_name,
                          mr.num_bufs_,
                          mr.buf_size_,
                          (void*)pool);

        q_packet_size += mr.buf_size_;
      }

      max_pkt_size = std::max(max_pkt_size, q_packet_size);
      uint32_t key = (intf.port_id_ << 16) | q.common_.id_;
      tx_q_map_[key] = q_backend;
    }

    HOLOSCAN_LOG_INFO("Max packet size needed with TX: {}", max_pkt_size);

    local_port_conf[intf.port_id_].txmode.offloads = 0;

    // Subtract eth headers since driver adds that on
    max_pkt_size = std::max(max_pkt_size, 64UL);
    local_port_conf[intf.port_id_].rxmode.mtu = std::min(max_pkt_size, (size_t)JUMBOFRAME_SIZE) -
      RTE_ETHER_CRC_LEN - RTE_ETHER_HDR_LEN;
    local_port_conf[intf.port_id_].rxmode.max_lro_pkt_size =
        local_port_conf[intf.port_id_].rxmode.mtu;

    HOLOSCAN_LOG_INFO("Setting port config for port {} mtu:{}",
                      intf.port_id_,
                      local_port_conf[intf.port_id_].rxmode.mtu);

    if (tx.accurate_send_) {
      setup_accurate_send_scheduling_mask();

      if (ret != 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", intf.port_id_);
        return;
      } else {
        if ((dev_info.tx_offload_capa & RTE_ETH_RX_OFFLOAD_TIMESTAMP) == 0) {
          HOLOSCAN_LOG_CRITICAL(
              "Accurate send scheduling enabled in config, but not supported by NIC!");
          return;
        } else {
          local_port_conf[intf.port_id_].txmode.offloads |= RTE_ETH_RX_OFFLOAD_TIMESTAMP;
        }
      }
    }

    if ((dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE) != 0) {
      local_port_conf[intf.port_id_].txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
    }

    local_port_conf[intf.port_id_].txmode.mq_mode = RTE_ETH_MQ_TX_NONE;
    local_port_conf[intf.port_id_].txmode.offloads |=
        RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM |
        RTE_ETH_TX_OFFLOAD_TCP_CKSUM | RTE_ETH_TX_OFFLOAD_MULTI_SEGS;


    HOLOSCAN_LOG_INFO("Initializing port {} with {} RX queues and {} TX queues...",
                      intf.port_id_,
                      intf.rx_.queues_.size(),
                      intf.tx_.queues_.size());

    ret = rte_eth_dev_configure(intf.port_id_,
                                intf.rx_.queues_.size(),
                                intf.tx_.queues_.size(),
                                &local_port_conf[intf.port_id_]);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Cannot configure device: err={}, str={}, port={}",
                            ret,
                            rte_strerror(ret),
                            intf.port_id_);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully configured ethdev");
    }

    ret =
        rte_eth_dev_adjust_nb_rx_tx_desc(intf.port_id_, &default_num_rx_desc, &default_num_tx_desc);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL(
          "Cannot adjust number of descriptors: err={}, port={}", ret, intf.port_id_);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully set descriptors to {}/{}",
        default_num_rx_desc, default_num_tx_desc);
    }

    rte_eth_macaddr_get(intf.port_id_, &conf_ports_eth_addr[intf.port_id_]);

    if (intf.flow_isolation_) {
      struct rte_flow_error error;
      ret = rte_flow_isolate(intf.port_id_, 1, &error);
      if (ret < 0) {
        HOLOSCAN_LOG_CRITICAL("Failed to set flow isolation");
      } else {
        HOLOSCAN_LOG_INFO("Port {} in isolation mode", intf.port_id_);
      }
    } else {
      HOLOSCAN_LOG_INFO("Port {} not in isolation mode", intf.port_id_);
    }

    for (const auto& q : rx.queues_) {
      // Assume one core for now
      auto socketid = rte_lcore_to_socket_id(strtol(q.common_.cpu_core_.c_str(), nullptr, 10));
      uint32_t key = (intf.port_id_ << 16) | q.common_.id_;
      auto qinfo = rx_q_map_[key];

      HOLOSCAN_LOG_INFO("Setting up port:{}, queue:{}, Num scatter:{} pool:{}",
                        intf.port_id_,
                        q.common_.id_,
                        q.common_.mrs_.size(),
                        (void*)qinfo->pools[0]);
      if (q.common_.mrs_.size() > 1) {
        ret = rte_eth_rx_queue_setup(intf.port_id_,
                                     q.common_.id_,
                                     default_num_rx_desc,
                                     socketid,
                                     &qinfo->rxconf_qsplit,
                                     NULL);
      } else {
        ret = rte_eth_rx_queue_setup(
            intf.port_id_, q.common_.id_, default_num_rx_desc, socketid, NULL, qinfo->pools[0]);
      }

      if (ret < 0) {
        HOLOSCAN_LOG_CRITICAL("rte_eth_rx_queue_setup: err={}, port={}", ret, intf.port_id_);
        return;
      } else {
        HOLOSCAN_LOG_INFO("Successfully setup RX port {} queue {}", intf.port_id_, q.common_.id_);
      }
    }

    struct rte_eth_txconf txq_conf;
    for (const auto& q : tx.queues_) {
      txq_conf = dev_info.default_txconf;
      txq_conf.offloads = local_port_conf[intf.port_id_].txmode.offloads;
      ret = rte_eth_tx_queue_setup(intf.port_id_,
                                   q.common_.id_,
                                   default_num_tx_desc,
                                   rte_eth_dev_socket_id(intf.port_id_),
                                   &txq_conf);
      if (ret < 0) {
        HOLOSCAN_LOG_CRITICAL("Queue setup error {}:{}, port={} caps={:x} set={:x}",
                              ret,
                              rte_strerror(ret),
                              intf.port_id_,
                              dev_info.tx_offload_capa,
                              local_port_conf[intf.port_id_].txmode.offloads);
        return;
      } else {
        HOLOSCAN_LOG_INFO("Successfully set up TX queue {}/{}", intf.port_id_, q.common_.id_);
      }
    }

    if (!intf.flow_isolation_) {
      HOLOSCAN_LOG_INFO("Enabling promiscuous mode for port {}", intf.port_id_);
      rte_eth_promiscuous_enable(intf.port_id_);
    } else {
      HOLOSCAN_LOG_INFO(
          "Not enabling promiscuous mode on port {} "
          "since flow isolation is enabled",
          intf.port_id_);
    }

    ret = rte_eth_dev_start(intf.port_id_);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Cannot start device err={}, port={}", ret, intf.port_id_);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully started port {}", intf.port_id_);
    }

    HOLOSCAN_LOG_INFO("Port {}, MAC address: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                      intf.port_id_,
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[0],
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[1],
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[2],
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[3],
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[4],
                      conf_ports_eth_addr[intf.port_id_].addr_bytes[5]);

    // Start flows
    int flow_num = 0;
    for (const auto& flow : rx.flows_) {
      HOLOSCAN_LOG_INFO("Adding RX flow {}", flow.name_);
      add_flow(intf.port_id_, flow);
    }

    apply_tx_offloads(intf.port_id_);
  }


  if (setup_pools_and_rings(max_rx_batch_size, max_tx_batch_size) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to set up pools and rings!");
    return;
  }
}

int DpdkMgr::setup_pools_and_rings(int max_rx_batch, int max_tx_batch) {
  HOLOSCAN_LOG_DEBUG("Setting up RX ring");
  rx_ring =
      rte_ring_create("RX_RING", 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
  if (rx_ring == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
    return -1;
  }

  auto num_rx_ptrs_bufs = (1UL << 13) - 1;
  HOLOSCAN_LOG_INFO("Setting up RX burst pool with {} batches of size {}",
                    num_rx_ptrs_bufs,
                    sizeof(void*) * max_rx_batch);
  rx_burst_buffer = rte_mempool_create("RX_BURST_POOL",
                                       num_rx_ptrs_bufs,
                                       sizeof(void*) * max_rx_batch,
                                       0,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       rte_socket_id(),
                                       0);
  if (rx_burst_buffer == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate RX burst pool!");
    return -1;
  }

  HOLOSCAN_LOG_INFO("Setting up RX burst pool with {} batches of size {}",
                    num_rx_ptrs_bufs,
                    sizeof(ExtraRxPacketInfo) * max_rx_batch);
  rx_flow_id_buffer = rte_mempool_create("RX_FLOWID_POOL",
                                       num_rx_ptrs_bufs,
                                       sizeof(ExtraRxPacketInfo) * max_rx_batch,
                                       0,
                                       0,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       rte_socket_id(),
                                       0);
  if (rx_flow_id_buffer == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate RX burst pool!");
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

      name = "TX_BURST_POOL_" + append;
      tx_burst_buffers[key] = rte_mempool_create(name.c_str(),
                                                 (1U << 7) - 1U,
                                                 sizeof(void*) * max_tx_batch,
                                                 0,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 rte_socket_id(),
                                                 0);
      if (tx_burst_buffers[key] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate TX message pool!");
        return -1;
      }

      HOLOSCAN_LOG_INFO("Setting up TX burst pool {} with {} pointers at {}",
                        name,
                        max_tx_batch,
                        (void*)tx_burst_buffers[key]);
    }
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

  return 0;
}

#define MAX_PATTERN_NUM 4
#define MAX_ACTION_NUM 3


// Taken from flow_block.c DPDK example */
struct rte_flow* DpdkMgr::add_flow(int port, const FlowConfig& cfg) {
  /* Declaring structs being used. 8< */
  struct rte_flow_attr attr;
  struct rte_flow_item pattern[MAX_PATTERN_NUM];
  struct rte_flow_action action[MAX_ACTION_NUM];
  struct rte_flow* flow = NULL;
  struct rte_flow_action_queue queue = {.index = cfg.action_.id_};
  struct rte_flow_action_mark  mark = {.id = cfg.id_};
  struct rte_flow_error error;
  struct rte_flow_item_udp udp_spec;
  struct rte_flow_item_udp udp_mask;
  struct rte_flow_item_ipv4  ip_spec;
  struct rte_flow_item_ipv4  ip_mask;
  struct rte_flow_item udp_item;
  int res;

  // HWS requires using a non-zero group, so we make a jump event to group 3 for all ethernet
  // packets
  {
    struct rte_flow_error jump_error;
    struct rte_flow_attr jump_attr{.group = 0, .ingress = 1};
    struct rte_flow_action_jump jump_v = {.group = 3};
    struct rte_flow_action jump_actions[] = {
      { .type = RTE_FLOW_ACTION_TYPE_JUMP, .conf = &jump_v},
      { .type = RTE_FLOW_ACTION_TYPE_END}
    };

    struct rte_flow_item jump_pattern[] = {
      { .type = RTE_FLOW_ITEM_TYPE_ETH, .spec = 0, .mask = 0},
      { .type = RTE_FLOW_ITEM_TYPE_END},
    };

    auto res = rte_flow_validate(port, &jump_attr, jump_pattern, jump_actions, &jump_error);
    if (!res) {
      struct rte_flow* flow = rte_flow_create(
          port, &jump_attr, jump_pattern, jump_actions, &jump_error);
      if (flow == nullptr) {
        HOLOSCAN_LOG_ERROR("rte_flow_create failed");
      }
    } else {
      HOLOSCAN_LOG_ERROR("Failed flow validation: {}", res);
    }
  }

  memset(pattern, 0, sizeof(pattern));
  memset(action, 0, sizeof(action));
  memset(&attr, 0, sizeof(struct rte_flow_attr));
  memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
  memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));

  action[0].type = RTE_FLOW_ACTION_TYPE_MARK;
  action[0].conf = &mark;
  action[1].type = RTE_FLOW_ACTION_TYPE_QUEUE;
  action[1].conf = &queue;
  action[2].type = RTE_FLOW_ACTION_TYPE_END;

  pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
  pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
  pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;

  if (cfg.match_.ipv4_len_ > 0) {
    ip_spec.hdr.total_length = htons(cfg.match_.ipv4_len_);
    ip_mask.hdr.total_length = 0xffff;
    pattern[1].spec = &ip_spec;
    pattern[1].mask = &ip_mask;
    HOLOSCAN_LOG_INFO("Adding IPv4 length match for {}", cfg.match_.ipv4_len_);
  }

  if (cfg.match_.udp_src_ > 0) {
    udp_spec.hdr.src_port = htons(cfg.match_.udp_src_);
    udp_spec.hdr.dst_port = htons(cfg.match_.udp_dst_);
    udp_spec.hdr.dgram_len = 0;
    udp_spec.hdr.dgram_cksum = 0;

    udp_mask.hdr.src_port = 0xffff;
    udp_mask.hdr.dst_port = 0xffff;
    udp_mask.hdr.dgram_len = 0;
    udp_mask.hdr.dgram_cksum = 0;

    udp_item.type = RTE_FLOW_ITEM_TYPE_UDP;
    udp_item.spec = &udp_spec;
    udp_item.mask = &udp_mask;
    udp_item.last = NULL;

    pattern[2] = udp_item;
    HOLOSCAN_LOG_INFO("Adding UDP port match for src/dst {}/{}",
      cfg.match_.udp_src_, cfg.match_.udp_dst_);
  }

  attr.ingress = 1;
  attr.priority = 0;
  attr.group = 3;

  pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

  flow = rte_flow_create(port, &attr, pattern, action, &error);
  return flow;
}

struct rte_flow* DpdkMgr::add_modify_flow_set(int port, int queue, const char* buf, int len,
                                              AdvNetDirection direction) {
  struct rte_flow_attr attr;
  struct rte_flow_item pattern[MAX_PATTERN_NUM];
  struct rte_flow_action action[MAX_ACTION_NUM];
  struct rte_flow* flow = NULL;
  struct rte_flow_action_modify_field mf;
  struct rte_flow_error error;
  struct rte_flow_item_eth eth;
  struct rte_flow_field_data src;
  struct rte_flow_field_data dst;

  int res;

  memset(pattern, 0, sizeof(pattern));
  memset(action, 0, sizeof(action));
  memset(&eth, 0, sizeof(struct rte_flow_item_eth));

  /* Set the rule attribute, only ingress packets will be checked. 8< */
  memset(&attr, 0, sizeof(struct rte_flow_attr));
  attr.ingress = (direction == AdvNetDirection::RX) ? 1 : 0;
  attr.egress = (direction == AdvNetDirection::TX) ? 1 : 0;

  // mf.operation = RTE_FLOW_MODIFY_SET;

  // mf.src.field      = RTE_FLOW_FIELD_VALUE;
  // mf.src.level      = 0;
  // mf.src.tag_index  = 0;
  // mf.src.type       = 0;
  // mf.src.class_id   = 0;
  // mf.src.offset     = 0;
  // memcpy(mf.src.value, buf, len / 8);
  // printf("%02x %02x %02x %02x %02x %02x %d\n", mf.src.value[0], mf.src.value[1], mf.src.value[2],
  // mf.src.value[3], mf.src.value[4], mf.src.value[5],len / 8);

  // mf.dst.field      = RTE_FLOW_FIELD_MAC_SRC;
  // mf.dst.level      = 0;
  // mf.dst.tag_index  = 0;
  // mf.src.type       = 0;
  // mf.src.class_id   = 0;
  // mf.src.offset     = 0;

  // mf.width = len;

  // action[0].type  = RTE_FLOW_ACTION_TYPE_MODIFY_FIELD;
  // action[0].conf  = &mf;
  // action[1].type  = RTE_FLOW_ACTION_TYPE_END;
  // pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
  // pattern[0].spec = &eth;
  // pattern[0].mask = &eth;
  // attr.priority = 0;

  // pattern[1].type = RTE_FLOW_ITEM_TYPE_END;

  struct rte_flow_action_set_mac sm;
  memcpy(&sm, buf, len / 8);
  action[0].type = RTE_FLOW_ACTION_TYPE_SET_MAC_SRC;
  action[0].conf = &sm;
  action[1].type = RTE_FLOW_ACTION_TYPE_END;
  pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
  pattern[0].spec = &eth;
  pattern[0].mask = &eth;
  attr.priority = 0;

  pattern[1].type = RTE_FLOW_ITEM_TYPE_END;

  res = rte_flow_validate(port, &attr, pattern, action, &error);
  if (!res) {
    flow = rte_flow_create(port, &attr, pattern, action, &error);
    return flow;
  }

  return nullptr;
}

void DpdkMgr::apply_tx_offloads(int port) {
  for (const auto& q : cfg_.ifs_[port].tx_.queues_) {
    for (const auto& off : q.common_.offloads_) {
      if (off == "tx_eth_src") {  // Offload Ethernet source copy
        HOLOSCAN_LOG_INFO("Applying {} offload for port {}", off, port);
        const auto mac_bytes = mac_addrs[port];
        add_modify_flow_set(port,
                            q.common_.id_,
                            reinterpret_cast<const char*>(&mac_bytes),
                            sizeof(mac_bytes) * 8,
                            AdvNetDirection::TX);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::PrintDpdkStats(int port) {
  struct rte_eth_stats eth_stats;
  int len, ret;

  rte_eth_stats_get(port, &eth_stats);
  HOLOSCAN_LOG_INFO("Port {}:", port);

  HOLOSCAN_LOG_INFO(" - Received packets:    {}", eth_stats.ipackets);
  HOLOSCAN_LOG_INFO(" - Transmit packets:    {}", eth_stats.opackets);
  HOLOSCAN_LOG_INFO(" - Received bytes:      {}", eth_stats.ibytes);
  HOLOSCAN_LOG_INFO(" - Transmit bytes:      {}", eth_stats.obytes);
  HOLOSCAN_LOG_INFO(" - Missed packets:      {}", eth_stats.imissed);
  HOLOSCAN_LOG_INFO(" - Errored packets:     {}", eth_stats.ierrors);
  HOLOSCAN_LOG_INFO(" - RX out of buffers:   {}", eth_stats.rx_nombuf);

  HOLOSCAN_LOG_INFO("   ** Extended Stats **");

  struct rte_eth_xstat *xstats;
  struct rte_eth_xstat_name *xstats_names;

  /* Clear screen and move to top left */
  len = rte_eth_xstats_get(port, NULL, 0);
  if (len < 0)
    rte_exit(EXIT_FAILURE, "rte_eth_xstats_get(%u) failed: %d", 0, len);
  xstats = (struct rte_eth_xstat *)calloc(len, sizeof(*xstats));
  if (xstats == NULL)
    rte_exit(EXIT_FAILURE, "Failed to calloc memory for xstats");
  ret = rte_eth_xstats_get(port, xstats, len);
  if (ret < 0 || ret > len) {
    free(xstats);
    rte_exit(EXIT_FAILURE, "rte_eth_xstats_get(%u) len%i failed: %d", 0, len, ret);
  }
  xstats_names = (struct rte_eth_xstat_name *)calloc(len, sizeof(*xstats_names));
  if (xstats_names == NULL) {
    free(xstats);
    rte_exit(EXIT_FAILURE, "Failed to calloc memory for xstats_names");
  }
  ret = rte_eth_xstats_get_names(port, xstats_names, len);
  if (ret < 0 || ret > len) {
    free(xstats);
    free(xstats_names);
    rte_exit(EXIT_FAILURE, "rte_eth_xstats_get_names(%u) len%i failed: %d", 0, len, ret);
  }
  for (int i = 0; i < len; i++) {
    if (xstats[i].value > 0)
    HOLOSCAN_LOG_INFO("      {}:\t\t{}", xstats_names[i].name, xstats[i].value);
  }

  free(xstats);
  free(xstats_names);
}

DpdkMgr::~DpdkMgr() {
}

bool DpdkMgr::validate_config() const {
  if (!ANOMgr::validate_config()) { return false; }

  HOLOSCAN_LOG_INFO("Config validated successfully");
  return true;
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::run() {
  int secondary_id = 0;
  int icore;

  HOLOSCAN_LOG_INFO("Starting advanced network workers");
  // determine the correct process types for input/output
  int (*rx_worker)(void*) = rx_core_worker;
  int (*tx_worker)(void*) = tx_core_worker;

  for (const auto& intf : cfg_.ifs_) {
    if (intf.rx_.queues_.size() > 0) {
      const auto& rx = intf.rx_;
      for (auto& q : rx.queues_) {
        // Dummy queue made to appease HWS. Don't launch worker
        if (q.common_.name_.find("UNUSED") == 0) {
          continue;
        }
        auto params = new RxWorkerParams;
        params->port = intf.port_id_;
        params->num_segs = q.common_.mrs_.size();
        params->ring = rx_ring;
        params->queue = q.common_.id_;
        params->burst_pool = rx_burst_buffer;
        params->flowid_pool = rx_flow_id_buffer;
        params->meta_pool = rx_meta;
        params->batch_size = q.common_.batch_size_;
        params->timeout_us = q.timeout_us_;
        rte_eal_remote_launch(
            rx_worker, (void*)params, strtol(q.common_.cpu_core_.c_str(), NULL, 10));
      }
    }

    if (intf.tx_.queues_.size() > 0) {
      const auto& tx = intf.tx_;
      for (auto& q : tx.queues_) {
        uint32_t key = (intf.port_id_ << 16) | q.common_.id_;
        auto params = new TxWorkerParams;
        //  params->hds    = q.common_.hds_ > 0;
        params->port = intf.port_id_;
        params->ring = tx_rings[key];
        params->queue = q.common_.id_;
        params->burst_pool = tx_burst_buffers[key];
        params->meta_pool = tx_meta;
        params->batch_size = q.common_.batch_size_;
        rte_eth_macaddr_get(intf.port_id_, &params->mac_addr);
        rte_eal_remote_launch(
            tx_worker, (void*)params, strtol(q.common_.cpu_core_.c_str(), NULL, 10));
      }
    }
  }

  HOLOSCAN_LOG_INFO("Done starting workers");
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::flush_packets(int port) {
  struct rte_mbuf* rx_mbuf;
  HOLOSCAN_LOG_INFO("Flushing packet on port {}", port);
  while (rte_eth_rx_burst(port, 0, &rx_mbuf, 1) != 0) { rte_pktmbuf_free(rx_mbuf); }
}


////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
int DpdkMgr::rx_core_worker(void* arg) {
  RxWorkerParams* tparams = (RxWorkerParams*)arg;
  struct rte_mbuf* rx_mbufs[DEFAULT_NUM_RX_BURST];
  int ret = 0;

  // In the future we may want to periodically update this if the CPU clock drifts
  uint64_t freq = rte_get_tsc_hz();
  uint64_t timeout_cycles = freq * (tparams->timeout_us/1e6);
  uint64_t last_cycles = rte_get_tsc_cycles();
  uint64_t total_pkts = 0;

  flush_packets(tparams->port);
  struct rte_mbuf* mbuf_arr[DEFAULT_NUM_RX_BURST];

  HOLOSCAN_LOG_INFO("Starting RX Core {}, port {}, queue {}, socket {}",
                    rte_lcore_id(),
                    tparams->port,
                    tparams->queue,
                    rte_socket_id());
  int nb_rx = 0;
  int to_copy = 0;
  int cur_pkt_in_batch = 0;
  //
  //  run loop
  //
  while (!force_quit.load()) {
    AdvNetBurstParams* burst;
    if (rte_mempool_get(tparams->meta_pool, reinterpret_cast<void**>(&burst)) < 0) {
      HOLOSCAN_LOG_ERROR("Processing function falling behind. No free buffers for metadata!");
      exit(1);
    }

    //  Queue ID for receiver to differentiate
    burst->hdr.hdr.q_id = tparams->queue;
    burst->hdr.hdr.port_id = tparams->port;
    burst->hdr.hdr.num_segs = tparams->num_segs;

    for (int seg = 0; seg < tparams->num_segs; seg++) {
      if (rte_mempool_get(tparams->burst_pool, reinterpret_cast<void**>(&burst->pkts[seg])) < 0) {
        HOLOSCAN_LOG_ERROR(
            "Processing function falling behind. No free RX bursts!");
        continue;
      }
    }

    if (rte_mempool_get(
          tparams->flowid_pool, reinterpret_cast<void**>(&burst->pkt_extra_info)) < 0) {
      HOLOSCAN_LOG_ERROR("Processing function falling behind. No free CPU buffers for packets!");
      continue;
    }

    ExtraRxPacketInfo *pkt_info = reinterpret_cast<ExtraRxPacketInfo*>(burst->pkt_extra_info);

    if (nb_rx > 0) {
      burst->hdr.hdr.num_pkts = nb_rx;

      // Copy non-scattered buffers
      memcpy(&burst->pkts[0][0], &mbuf_arr[to_copy], sizeof(rte_mbuf*) * nb_rx);

      for (int p = 0; p < nb_rx; p++) {
        if (mbuf_arr[to_copy + p]->ol_flags & RTE_MBUF_F_RX_FDIR_ID) {
          pkt_info[p].flow_id = mbuf_arr[to_copy + p]->hash.fdir.hi;
        } else {
          pkt_info[p].flow_id = 0;
        }
      }

      if (tparams->num_segs > 1) {  // Extra work when buffers are scattered
        for (int p = 0; p < nb_rx; p++) {
          struct rte_mbuf* mbuf = mbuf_arr[to_copy + p];
          for (int seg = 1; seg < tparams->num_segs; seg++) {
            mbuf = mbuf->next;
            burst->pkts[seg][p] = mbuf;
          }
        }

        cur_pkt_in_batch += nb_rx;
      }

      nb_rx = 0;
    } else {
      burst->hdr.hdr.num_pkts = 0;
    }

    // DPDK on some ARM platforms requires that you always pass nb_pkts as a number divisible
    // by 4. If you pass something other than that, you get undefined results and will end up
    // running out of buffers.
    do {
      int burst_size = std::min((uint32_t)DEFAULT_NUM_RX_BURST,
                                (uint32_t)(tparams->batch_size - burst->hdr.hdr.num_pkts));

      nb_rx = rte_eth_rx_burst(tparams->port,
                               tparams->queue,
                               reinterpret_cast<rte_mbuf**>(&mbuf_arr[0]),
                               DEFAULT_NUM_RX_BURST);

      if (nb_rx == 0) {
        if (burst->hdr.hdr.num_pkts > 0 && timeout_cycles > 0) {
          const auto cur_cycles = rte_get_tsc_cycles();

          // We hit our timeout. Send the partial batch immediately
          if ((cur_cycles - last_cycles) > timeout_cycles) {
            cur_pkt_in_batch = 0;
            rte_ring_enqueue(tparams->ring, reinterpret_cast<void*>(burst));
            last_cycles = cur_cycles;
            break;
          }
        }

        continue;
      }

      to_copy = std::min(nb_rx, (int)(tparams->batch_size - burst->hdr.hdr.num_pkts));
      memcpy(&burst->pkts[0][burst->hdr.hdr.num_pkts], &mbuf_arr, sizeof(rte_mbuf*) * to_copy);

      for (int p = 0; p < to_copy; p++) {
        if (mbuf_arr[p]->ol_flags & RTE_MBUF_F_RX_FDIR_ID) {
          pkt_info[burst->hdr.hdr.num_pkts + p].flow_id = mbuf_arr[p]->hash.fdir.hi;
        } else {
          pkt_info[burst->hdr.hdr.num_pkts + p].flow_id = 0;
        }
      }

      if (tparams->num_segs > 1) {  // Extra work when buffers are scattered
        for (int p = 0; p < to_copy; p++) {
          struct rte_mbuf* mbuf = mbuf_arr[p];
          for (int seg = 1; seg < tparams->num_segs; seg++) {
            mbuf = mbuf->next;
            burst->pkts[seg][cur_pkt_in_batch + p] = mbuf;
          }
        }

        cur_pkt_in_batch += to_copy;
      }

      burst->hdr.hdr.num_pkts += to_copy;
      total_pkts += nb_rx;
      nb_rx -= to_copy;

      if (burst->hdr.hdr.num_pkts == tparams->batch_size) {
        rte_ring_enqueue(tparams->ring, reinterpret_cast<void*>(burst));
        cur_pkt_in_batch = 0;
        last_cycles = rte_get_tsc_cycles();
        break;
      } else if (timeout_cycles > 0) {
        const auto cur_cycles = rte_get_tsc_cycles();

        // We hit our timeout. Send the partial batch immediately
        if ((cur_cycles - last_cycles) > timeout_cycles) {
          rte_ring_enqueue(tparams->ring, reinterpret_cast<void*>(burst));
          cur_pkt_in_batch = 0;
          last_cycles = cur_cycles;
          break;
        }
      }
    } while (!force_quit.load());
  }

  HOLOSCAN_LOG_INFO("Total packets received by application (port/queue {}/{}): {}",
                     tparams->port,
                     tparams->queue,
                     total_pkts);

  return 0;
}

int DpdkMgr::tx_core_worker(void* arg) {
  TxWorkerParams* tparams = (TxWorkerParams*)arg;
  uint64_t seq;
  uint64_t ttl_pkts_tx = 0;
  AdvNetBurstParams* msg;
  int64_t bursts = 0;

  HOLOSCAN_LOG_INFO("Starting TX Core {}, port {}, queue {} socket {} using burst pool {} ring {}",
                    rte_lcore_id(),
                    tparams->port,
                    tparams->queue,
                    rte_socket_id(),
                    (void*)tparams->burst_pool,
                    (void*)tparams->ring);

  while (!force_quit.load()) {
    if (rte_ring_dequeue(tparams->ring, reinterpret_cast<void**>(&msg)) != 0) { continue; }

    // Scatter mode needs to chain all the buffers
    if (msg->hdr.hdr.num_segs > 1) {
      for (size_t p = 0; p < msg->hdr.hdr.num_pkts; p++) {
        for (int seg = 0; seg < msg->hdr.hdr.num_segs; seg++) {
          auto* mbuf = reinterpret_cast<struct rte_mbuf*>(msg->pkts[seg][p]);
          mbuf->next = reinterpret_cast<struct rte_mbuf*>(msg->pkts[seg + 1][p]);
        }

        reinterpret_cast<struct rte_mbuf*>(msg->pkts[0][p])->nb_segs = msg->hdr.hdr.num_segs;
      }
    }

    //     if (msg->pkts[0] != nullptr) {
    //       for (size_t p = 0; p < msg->hdr.hdr.num_pkts; p++) {
    //         auto *mbuf = reinterpret_cast<rte_mbuf*>(msg->pkts[0][p]);
    //         auto *pkt  = rte_pktmbuf_mtod(mbuf, uint8_t*);
    // #pragma GCC diagnostic push
    // #pragma GCC diagnostic ignored "-Waddress-of-packed-member"
    //         rte_ether_addr_copy(&tparams->mac_addr, reinterpret_cast<rte_ether_addr *>(pkt + 6));
    // #pragma GCC diagnostic pop
    //       }
    //     }

    auto pkts_to_transmit = static_cast<int64_t>(msg->hdr.hdr.num_pkts);

    size_t pkts_tx = 0;
    while (pkts_tx != msg->hdr.hdr.num_pkts && !force_quit.load()) {
      auto to_send = static_cast<uint16_t>(
          std::min(static_cast<size_t>(DEFAULT_NUM_TX_BURST), msg->hdr.hdr.num_pkts - pkts_tx));

      // CPU-only or HDS mode
      int tx;
      tx = rte_eth_tx_burst(tparams->port,
                            tparams->queue,
                            reinterpret_cast<rte_mbuf**>(&msg->pkts[0][pkts_tx]),
                            to_send);

      pkts_tx += tx;
    }

    ttl_pkts_tx += pkts_tx;

    for (int seg = 0; seg < msg->hdr.hdr.num_segs; seg++) {
      rte_mempool_put(tparams->burst_pool, static_cast<void*>(msg->pkts[seg]));
    }

    rte_mempool_put(tparams->meta_pool, msg);
    bursts++;
  }

  HOLOSCAN_LOG_INFO("Total packets transmitted by application (port/queue {}/{}): {}",
                     tparams->port,
                     tparams->queue,
                     ttl_pkts_tx);

  return 0;
}

/* ANO interface implementations */
void* DpdkMgr::get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->pkts[seg][idx]), void*);
}

void* DpdkMgr::get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx]), void*);
}

uint16_t DpdkMgr::get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
  return reinterpret_cast<rte_mbuf*>(burst->pkts[seg][idx])->data_len;
}

uint16_t DpdkMgr::get_pkt_len(AdvNetBurstParams* burst, int idx) {
  return reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx])->pkt_len;
}

uint16_t DpdkMgr::get_pkt_flow_id(AdvNetBurstParams* burst, int idx) {
  const ExtraRxPacketInfo* info = reinterpret_cast<ExtraRxPacketInfo*>(burst->pkt_extra_info);
  return info[idx].flow_id;
}

AdvNetStatus DpdkMgr::set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp) {
  reinterpret_cast<struct rte_mbuf**>(burst->pkts[0])[idx]->ol_flags |= timestamp_mask_;
  *RTE_MBUF_DYNFIELD(
      reinterpret_cast<rte_mbuf**>(burst->pkts[0])[idx], timestamp_offset_, uint64_t*) = timestamp;

  return AdvNetStatus::SUCCESS;
}

void* DpdkMgr::get_pkt_extra_info(AdvNetBurstParams* burst, int idx) {
  return nullptr;
}

AdvNetStatus DpdkMgr::get_tx_pkt_burst(AdvNetBurstParams* burst) {
  const uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
  const auto& q = tx_q_map_[key];

  const auto burst_pool = tx_burst_buffers.find(key);
  if (burst_pool == tx_burst_buffers.end()) {
    HOLOSCAN_LOG_ERROR("Failed to look up burst pool name for port {} queue {}",
                       burst->hdr.hdr.port_id,
                       burst->hdr.hdr.q_id);
    return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  }

  for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
    if (rte_mempool_get(burst_pool->second, reinterpret_cast<void**>(&burst->pkts[seg])) != 0) {
      return AdvNetStatus::NO_FREE_BURST_BUFFERS;
    }

    if (rte_pktmbuf_alloc_bulk(q->pools[seg],
                               reinterpret_cast<rte_mbuf**>(burst->pkts[seg]),
                               static_cast<int>(burst->hdr.hdr.num_pkts)) != 0) {
      rte_mempool_put(burst_pool->second, reinterpret_cast<void*>(burst->pkts[seg]));
      return AdvNetStatus::NO_FREE_PACKET_BUFFERS;
    }
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);
  memcpy(reinterpret_cast<void*>(&mbuf_data->eth.dst_addr),
         reinterpret_cast<void*>(dst_addr),
         sizeof(mbuf_data->eth.dst_addr));

  mbuf_data->eth.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                   unsigned int src_host, unsigned int dst_host) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);
  mbuf_data->ip.next_proto_id = proto;
  mbuf_data->ip.ihl = 5;
  mbuf_data->ip.total_length = rte_cpu_to_be_16(sizeof(mbuf_data->ip) + ip_len);
  mbuf_data->ip.version = 4;
  mbuf_data->ip.src_addr = htonl(src_host);
  mbuf_data->ip.dst_addr = htonl(dst_host);
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                  uint16_t dst_port) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);

  mbuf_data->udp.dgram_cksum = 0;
  mbuf_data->udp.src_port = htons(src_port);
  mbuf_data->udp.dst_port = htons(dst_port);
  mbuf_data->udp.dgram_len = htons(udp_len + sizeof(mbuf_data->udp));
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->pkts[0][idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);

  rte_memcpy(mbuf_data->payload, data, len);
  return AdvNetStatus::SUCCESS;
}

bool DpdkMgr::tx_burst_available(AdvNetBurstParams* burst) {
  const uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
  const auto& q = tx_q_map_[key];

  for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
    if (rte_mempool_avail_count(q->pools[seg]) < burst->hdr.hdr.num_pkts * 2) { return false; }
  }

  return true;
}

AdvNetStatus DpdkMgr::set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                   const std::initializer_list<int>& lens) {
  uint32_t ttl_len = 0;
  for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
    reinterpret_cast<rte_mbuf**>(burst->pkts[seg])[idx]->data_len = *(lens.begin() + seg);
    ttl_len += *(lens.begin() + seg);
  }

  reinterpret_cast<rte_mbuf**>(burst->pkts[0])[idx]->pkt_len = ttl_len;

  return AdvNetStatus::SUCCESS;
}

void DpdkMgr::free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) {
  rte_pktmbuf_free_seg(reinterpret_cast<rte_mbuf**>(burst->pkts[seg])[pkt]);
}

void DpdkMgr::free_all_seg_pkts(AdvNetBurstParams* burst, int seg) {
  for (int p = 0; p < burst->hdr.hdr.num_pkts; p++) {
    rte_pktmbuf_free_seg(reinterpret_cast<rte_mbuf**>(burst->pkts[seg])[p]);
  }
}

void DpdkMgr::free_pkt(AdvNetBurstParams* burst, int pkt) {
  rte_pktmbuf_free(reinterpret_cast<rte_mbuf**>(burst->pkts[0])[pkt]);
}

void DpdkMgr::free_all_pkts(AdvNetBurstParams* burst) {
  for (int p = 0; p < burst->hdr.hdr.num_pkts; p++) {
    rte_pktmbuf_free(reinterpret_cast<rte_mbuf**>(burst->pkts[0])[p]);
  }
}

void DpdkMgr::free_rx_burst(AdvNetBurstParams* burst) {
  rte_mempool_put(rx_flow_id_buffer, (void*)burst->pkt_extra_info);
  rte_mempool_put(rx_meta, burst);
  for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
    rte_mempool_put(rx_burst_buffer, (void*)burst->pkts[seg]);
  }
}

void DpdkMgr::free_tx_burst(AdvNetBurstParams* burst) {
  const uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
  const auto burst_pool = tx_burst_buffers.find(key);

  for (int seg = 0; seg < burst->hdr.hdr.num_segs; seg++) {
    rte_mempool_put(burst_pool->second, (void*)burst->pkts[seg]);
  }
}

std::optional<uint16_t> DpdkMgr::get_port_from_ifname(const std::string& name) {
  uint16_t port;
  auto ret = rte_eth_dev_get_port_by_name(name.c_str(), &port);
  if (ret < 0) { return {}; }

  return port;
}

AdvNetStatus DpdkMgr::get_rx_burst(AdvNetBurstParams** burst) {
  if (rte_ring_dequeue(rx_ring, reinterpret_cast<void**>(burst)) < 0) {
    return AdvNetStatus::NOT_READY;
  }

  return AdvNetStatus::SUCCESS;
}

void DpdkMgr::free_rx_meta(AdvNetBurstParams* burst) {
  rte_mempool_put(rx_meta, burst);
}

void DpdkMgr::free_tx_meta(AdvNetBurstParams* burst) {
  rte_mempool_put(tx_meta, burst);
}

AdvNetStatus DpdkMgr::get_tx_meta_buf(AdvNetBurstParams** burst) {
  if (rte_mempool_get(tx_meta, reinterpret_cast<void**>(burst)) != 0) {
    HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor");
    return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DpdkMgr::send_tx_burst(AdvNetBurstParams* burst) {
  uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
  const auto ring = tx_rings.find(key);

  if (ring == tx_rings.end()) {
    HOLOSCAN_LOG_ERROR("Invalid port/queue combination in send_tx_burst: {}/{}",
                       burst->hdr.hdr.port_id,
                       burst->hdr.hdr.q_id);
    return AdvNetStatus::INVALID_PARAMETER;
  }

  if (rte_ring_enqueue(ring->second, reinterpret_cast<void*>(burst)) != 0) {
    free_tx_burst(burst);
    free_tx_meta(burst);
    HOLOSCAN_LOG_CRITICAL("Failed to enqueue TX work");
    return AdvNetStatus::NO_SPACE_AVAILABLE;
  }

  return AdvNetStatus::SUCCESS;
}

void DpdkMgr::shutdown() {
  HOLOSCAN_LOG_INFO("DPDK ANO shutdown called {}", num_init);
  if (--num_init == 0) {
    int portid;
    RTE_ETH_FOREACH_DEV(portid) {
      PrintDpdkStats(portid);
    }

    HOLOSCAN_LOG_INFO("ANO DPDK manager shutting down");
    force_quit.store(true);
  }
}

void DpdkMgr::print_stats() {
  int portid;
  RTE_ETH_FOREACH_DEV(portid) {
    PrintDpdkStats(portid);
  }
}

uint64_t DpdkMgr::get_burst_tot_byte(AdvNetBurstParams* burst) {
  return 0;
}

AdvNetBurstParams* DpdkMgr::create_burst_params() {
  return new AdvNetBurstParams();
}

};  // namespace holoscan::ops
