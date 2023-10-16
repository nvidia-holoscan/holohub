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

DpdkMgr dpdk_mgr{};

std::atomic<bool> force_quit = false;

struct TxWorkerParams {
  int port;
  int queue;
  uint32_t batch_size;
  struct rte_ring *ring;
  struct rte_mempool *meta_pool;
  struct rte_mempool *burst_pool;
  struct rte_ether_addr mac_addr;
};

struct RxWorkerParams {
  int port;
  int queue;
  uint32_t batch_size;
  struct rte_ring *ring;
  struct rte_mempool *burst_pool;
  struct rte_mempool *meta_pool;
  uint64_t rx_pkts = 0;
  bool gpu_direct;
  bool hds;
};


struct DPDKQueueConfig {
  struct rte_mempool *pools[DpdkMgr::BUFFER_SPLIT_SEGS];
  struct rte_eth_rxconf rxconf_qsplit;
  union  rte_eth_rxseg  rx_useg[DpdkMgr::BUFFER_SPLIT_SEGS] = {};
};



////////////////////////////////////////////////////////////////////////////////
///
///  \brief Init
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::SetConfigAndInitialize(const AdvNetConfigYaml &cfg) {
  if (!initialized) {
    cfg_ = cfg;
    cpu_set_t mask;
    long nproc, i;

    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&DpdkMgr::Initialize, this);
    t.join();
    Run();
  }
}

std::optional<struct rte_pktmbuf_extmem> DpdkMgr::AllocateGpuPktMbuf(
    int port_id,
    uint16_t pkt_size,
    int num_mbufs,
    int gpu_dev) {
  struct rte_pktmbuf_extmem ext_mem;
  auto target_el_size = pkt_size + RTE_PKTMBUF_HEADROOM;
  ext_mem.elt_size = ((target_el_size + 3) / 4) * 4;

  struct rte_eth_dev_info dev_info;
  int ret = rte_eth_dev_info_get(port_id, &dev_info);
  if (ret != 0) {
    HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", port_id);
    return std::nullopt;
  }

  ext_mem.buf_len = RTE_ALIGN_CEIL(static_cast<size_t>(num_mbufs) *
                                    static_cast<size_t>(ext_mem.elt_size), GPU_PAGE_SIZE);
  HOLOSCAN_LOG_INFO("Allocated {} buffers elt_size={} totaling {} bytes of GPU memory for packets",
        num_mbufs, ext_mem.elt_size, ext_mem.buf_len);
  ext_mem.buf_iova = RTE_BAD_IOVA;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ext_mem.buf_ptr = rte_gpu_mem_alloc(gpu_dev, ext_mem.buf_len, CPU_PAGE_SIZE);
#pragma GCC diagnostic pop
  if (ext_mem.buf_ptr == NULL) {
    HOLOSCAN_LOG_CRITICAL("Could not allocate {:.2f}MB of GPU memory", ext_mem.buf_len/1e6);
    return std::nullopt;
  } else {
    HOLOSCAN_LOG_INFO("Allocated {:.2f}MB on GPU", ext_mem.buf_len/1e6);
  }

  ret = rte_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova,
        GPU_PAGE_SIZE);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Unable to register addr {}, ret {}", ext_mem.buf_ptr, ret);
    return std::nullopt;
  } else {
    HOLOSCAN_LOG_INFO("Successfully registered external memory");
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  ret = rte_dev_dma_map(dev_info.device, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len);
#pragma GCC diagnostic pop
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Could not DMA map EXT memory: {} err={}", ret, rte_errno);
    return std::nullopt;
  }

  return std::make_optional(ext_mem);
}



void DpdkMgr::Initialize() {
  int ret;
  uint16_t portid;

  static struct rte_eth_conf conf_eth_port = {
    .rxmode = {
      .mq_mode = RTE_ETH_MQ_RX_RSS,
      .offloads = RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT,  // Required by buffer split feature
    },
    .txmode = {
      .mq_mode = RTE_ETH_MQ_TX_NONE,
      .offloads = RTE_ETH_TX_OFFLOAD_MULTI_SEGS,
    },
    .rx_adv_conf = {
      .rss_conf = {
        .rss_key = NULL,
        .rss_hf = RTE_ETH_RSS_IP
      },
    },
  };

  for (auto &conf : local_port_conf) {
    conf = conf_eth_port;
  }

  /* Initialize DPDK params */
  constexpr int max_nargs = 32;
  constexpr int max_arg_size = 64;
  char **_argv;
  _argv = (char**)malloc(sizeof(char*) * max_nargs);
  for (int i = 0; i < max_nargs; i++) {
    _argv[i] = (char*)malloc(max_arg_size);
  }

  int arg = 0;
  std::string cores = std::to_string(cfg_.common_.master_core_) + ",";  // Master core must be first
  std::set<std::string> ifs;
  std::set<std::string> gpu_bdfs;
  std::unordered_map<uint16_t, std::pair<uint16_t, uint16_t>> port_q_num;
  std::unordered_map<uint16_t, std::string> port_id_to_name;

  // Get GPU PCIe BDFs since they're needed to pass to DPDK
  for (const auto &rx : cfg_.rx_) {
    ifs.emplace(rx.if_name_);
    for (const auto &q : rx.queues_) {
      cores += q.common_.cpu_cores_ + ",";

      if (q.common_.gpu_direct_) {
        char gpu_bdf[32];
        if (cudaDeviceGetPCIBusId(gpu_bdf, sizeof(gpu_bdf), q.common_.gpu_dev_) != cudaSuccess) {
          HOLOSCAN_LOG_CRITICAL("Cannot get GPU BDF for device ID {}", q.common_.gpu_dev_);
          return;
        }

        gpu_bdfs.insert(std::string(gpu_bdf));
        HOLOSCAN_LOG_INFO("Found GPU BDF {} for device ID {}", gpu_bdf, q.common_.gpu_dev_);
      }
    }
  }

  for (const auto &tx : cfg_.tx_) {
    ifs.emplace(tx.if_name_);
    for (const auto &q : tx.queues_) {
      cores += q.common_.cpu_cores_ + ",";

      if (q.common_.gpu_direct_) {
        char gpu_bdf[32];
        if (cudaDeviceGetPCIBusId(gpu_bdf, sizeof(gpu_bdf), q.common_.gpu_dev_) != cudaSuccess) {
          HOLOSCAN_LOG_CRITICAL("Cannot get GPU BDF for device ID {}", q.common_.gpu_dev_);
          return;
        }

        gpu_bdfs.insert(std::string(gpu_bdf));
        HOLOSCAN_LOG_INFO("Found GPU BDF {} for device ID {}", gpu_bdf, q.common_.gpu_dev_);
      }
    }
  }

  cores = cores.substr(0, cores.size()-1);
  // Get a unique set of interfaces
  num_ports = ifs.size();
  HOLOSCAN_LOG_INFO("Attempting to use {} ports for high-speed network", num_ports);

  strncpy(_argv[arg++], "adv_net_operator", max_arg_size - 1);
  strncpy(_argv[arg++], "-l", max_arg_size - 1);
  strncpy(_argv[arg++], cores.c_str(), max_arg_size - 1);

  //  strncpy(_argv[arg++], "--log-level=99", max_arg_size - 1);
  //  strncpy(_argv[arg++], "--log-level=pmd.net.mlx5:8", max_arg_size - 1);
  for (const auto &name : ifs) {
    strncpy(_argv[arg++], "-a", max_arg_size - 1);
    strncpy(_argv[arg++],
    (name + std::string(",txq_inline_max=0,dv_flow_en=1")).c_str(), max_arg_size - 1);
  }

  // Add GPU PCIe devices
  for (const auto &gpu : gpu_bdfs) {
    strncpy(_argv[arg++], "-a", max_arg_size - 1);
    strncpy(_argv[arg++], gpu.c_str(), max_arg_size - 1);
  }

  _argv[arg] = nullptr;
  std::string dpdk_args = "";
  for (int ac = 0; ac < arg; ac++) {
    dpdk_args += std::string(_argv[ac]) + " ";
  }

  HOLOSCAN_LOG_INFO("DPDK EAL arguments: {}", dpdk_args);

  ret = rte_eal_init(arg, _argv);
  if (ret < 0) {
    HOLOSCAN_LOG_CRITICAL("Invalid EAL arguments: {}", rte_errno);
    return;
  }

  for (int i = 0; i < num_ports; i++) {
    rte_eth_macaddr_get(i, &mac_addrs[i]);
  }

  // Build name to id mapping
  for (auto &rx : cfg_.rx_) {
    ret = rte_eth_dev_get_port_by_name(rx.if_name_.c_str(), &rx.port_id_);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port number for {}", rx.if_name_.c_str());
      return;
    }

    port_q_num[rx.port_id_] = {rx.queues_.size(), 0};
    port_id_to_name[rx.port_id_] = rx.if_name_;
    rx.empty = false;
  }

  for (auto &tx : cfg_.tx_) {
    ret = rte_eth_dev_get_port_by_name(tx.if_name_.c_str(), &tx.port_id_);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port number for {}", tx.if_name_.c_str());
      return;
    }

    if (port_q_num.find(tx.port_id_) == port_q_num.end()) {
      port_q_num[tx.port_id_] = {0, tx.queues_.size()};
    } else {
      port_q_num[tx.port_id_].second = tx.queues_.size();
    }
    port_id_to_name[tx.port_id_] = tx.if_name_;
    tx.empty = false;
  }

  HOLOSCAN_LOG_INFO("DPDK init -- RX: {} TX: {}", cfg_.rx_.size() > 0 ? "ENABLED" : "DISABLED",
                                                  cfg_.tx_.size() > 0 ? "ENABLED" : "DISABLED");

  // Create any missing queues default queues
  for (const auto &[port, queues] : port_q_num) {
    if (queues.first == 0) {
      HOLOSCAN_LOG_INFO("Creating default queue for port {} receive", port);
      RxQueueConfig q = {.common_ = {"Default", 0, 0, false, 0, "0", 1518, 32767, 1, nullptr}};
      AdvNetRxConfig rxcfg;
      rxcfg.if_name_ = port_id_to_name[port];
      rxcfg.port_id_ = port;
      rxcfg.empty = true;
      rxcfg.queues_.emplace_back(q);
      cfg_.rx_.emplace_back(rxcfg);
      port_q_num[port].first = 1;
    } else if (queues.second == 0) {
      HOLOSCAN_LOG_INFO("Creating default queue for port {} transmit", port);
      TxQueueConfig q = {.common_ = {"Default", 0, 0, false, 0, "0", 1518, 32767, 1, nullptr}};
      AdvNetTxConfig txcfg;
      txcfg.if_name_ = port_id_to_name[port];
      txcfg.port_id_ = port;
      txcfg.empty = true;
      txcfg.queues_.emplace_back(q);
      cfg_.tx_.emplace_back(txcfg);
      port_q_num[port].second = 1;
    }
  }

  // Queue setup
  int max_rx_batch_size = 0;
  for (auto &rx : cfg_.rx_) {
    int max_pkt_size = 0;
    ret = rte_eth_dev_get_port_by_name(rx.if_name_.c_str(), &rx.port_id_);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port number for {}", rx.if_name_.c_str());
      return;
    }

    for (auto &q : rx.queues_) {
      HOLOSCAN_LOG_INFO("Configuring RX queue: {} ({}) on port {}",
            q.common_.name_, q.common_.id_, rx.port_id_);
      q.common_.backend_config_ = new DPDKQueueConfig;
      auto q_backend = static_cast<DPDKQueueConfig *>(q.common_.backend_config_);

      std::string append = "_P" + std::to_string(rx.port_id_) + "_Q" +
          std::to_string(q.common_.id_);
      auto rx_mbufs = q.common_.num_concurrent_batches_* q.common_.batch_size_;
      max_rx_batch_size = std::max(max_rx_batch_size, q.common_.batch_size_);
      max_pkt_size   = std::max(max_pkt_size, q.common_.max_packet_size_);

      if (q.common_.gpu_direct_) {
        auto ext_mem = AllocateGpuPktMbuf(rx.port_id_,
                                          q.common_.max_packet_size_ - q.common_.hds_,
                                          rx_mbufs,
                                          q.common_.gpu_dev_);
        if (!ext_mem) {
          HOLOSCAN_LOG_ERROR("Failed to allocate GPU packet pool");
          return;
        }

        if (q.common_.hds_ == 0) {
          std::string gpu_name = std::string("RX_GPU_POOL") + append;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
          q_backend->pools[0] = rte_pktmbuf_pool_create_extbuf(gpu_name.c_str(), rx_mbufs,
              0, 0, ext_mem->elt_size, rte_socket_id(), std::addressof(*ext_mem), 1);
    #pragma GCC diagnostic pop
          if (q_backend->pools[0] == NULL) {
            HOLOSCAN_LOG_CRITICAL("Could not create EXT memory mempool");
            return;
          }

          HOLOSCAN_LOG_INFO("Created GPU mempool for GPUDirect: {} mbufs={} elsize={} ptr={}",
              gpu_name, rx_mbufs, ext_mem->elt_size, (void*)q_backend->pools[0]);
        } else {
          HOLOSCAN_LOG_INFO(
              "Enabling header-data split on RX with split point of {} and GPU payload size {}",
                q.common_.hds_, ext_mem->elt_size);

          std::string gpu_name = std::string("RX_GPU_POOL") + append;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
          q_backend->pools[1] = rte_pktmbuf_pool_create_extbuf(gpu_name.c_str(), rx_mbufs,
              0, 0, ext_mem->elt_size, rte_socket_id(), std::addressof(*ext_mem), 1);
    #pragma GCC diagnostic pop
          if (q_backend->pools[1] == NULL) {
            HOLOSCAN_LOG_CRITICAL("Could not create EXT memory mempool");
            return;
          }

          HOLOSCAN_LOG_INFO("Created GPU mempool for RX HDS: {} mbufs={} elsize={} ptr={}",
              gpu_name, rx_mbufs, ext_mem->elt_size, (void*)q_backend->pools[1]);

          auto cpu_name = std::string("RX_CPU_POOL") + append;
          q_backend->pools[0] = rte_pktmbuf_pool_create(cpu_name.c_str(),
              rx_mbufs,  MEMPOOL_CACHE_SIZE, 0, q.common_.hds_ + RTE_PKTMBUF_HEADROOM,
                  rte_socket_id());
          if (!q_backend->pools[0]) {
            HOLOSCAN_LOG_CRITICAL("Could not create sysmem mempool {} buffer split: {}",
                cpu_name, rte_errno);
            return;
          }

          HOLOSCAN_LOG_INFO("Created CPU mempool for RX HDS: {} mbufs={} elsize={} ptr={}",
            cpu_name, rx_mbufs, q.common_.hds_ + RTE_PKTMBUF_HEADROOM, (void*)q_backend->pools[0]);

          struct rte_eth_dev_info dev_info;
          int ret = rte_eth_dev_info_get(rx.port_id_, &dev_info);
          if (ret != 0) {
            HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", rx.port_id_);
            return;
          }

          memcpy(&q_backend->rxconf_qsplit, &dev_info.default_rxconf,
              sizeof(q_backend->rxconf_qsplit));

          q_backend->rxconf_qsplit.offloads =
              RTE_ETH_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
          q_backend->rxconf_qsplit.rx_nseg  = BUFFER_SPLIT_SEGS;
          q_backend->rxconf_qsplit.rx_seg   = q_backend->rx_useg;

          struct rte_eth_rxseg_split *rx_seg;
          rx_seg = &q_backend->rx_useg[0].split;
          rx_seg->mp = q_backend->pools[0];
          rx_seg->length = q.common_.hds_;
          rx_seg->offset = 0;

          rx_seg = &q_backend->rx_useg[1].split;
          rx_seg->mp = q_backend->pools[1];
          rx_seg->length = 0;
          rx_seg->offset = 0;

          local_port_conf[rx.port_id_].rxmode.offloads |=
              RTE_ETH_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
        }
      } else {
        /* Create the mbuf pools. */
        auto pkt_size = q.common_.max_packet_size_ + RTE_PKTMBUF_HEADROOM;
        auto name = std::string("RX_CPU_POOL") + append;
        q_backend->pools[0] = rte_pktmbuf_pool_create(name.c_str(),
            rx_mbufs, MEMPOOL_CACHE_SIZE, 0, pkt_size, rte_socket_id());
        if (q_backend->pools[0] == NULL) {
          HOLOSCAN_LOG_CRITICAL("Cannot init mbuf pool");
          return;
        }

        HOLOSCAN_LOG_INFO("Created RX pool {} with packet size {} bytes and {} mbufs at {}",
            name, pkt_size, rx_mbufs, (void*)q_backend->pools[0]);
      }
    }

    HOLOSCAN_LOG_INFO("Setting port config for port {} mtu:{}", rx.port_id_, max_pkt_size);
    local_port_conf[rx.port_id_].rxmode.offloads |= RTE_ETH_RX_OFFLOAD_CHECKSUM;

    // Subtract eth headers since driver adds that on
    local_port_conf[rx.port_id_].rxmode.mtu = max_pkt_size - RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN;
    local_port_conf[rx.port_id_].rxmode.max_lro_pkt_size =
        max_pkt_size - RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN;
  }


  // For now make a single queue. Support more sophisticated TX on next release
  int max_tx_batch_size = 0;
  for (auto &tx : cfg_.tx_) {
    ret = rte_eth_dev_get_port_by_name(tx.if_name_.c_str(), &tx.port_id_);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port number for {}", tx.if_name_.c_str());
      return;
    }

    HOLOSCAN_LOG_INFO("Using port {} for TX", tx.port_id_);

    for (auto &q : tx.queues_) {
      max_tx_batch_size = std::max(max_tx_batch_size, q.common_.batch_size_);
      q.common_.backend_config_ = new DPDKQueueConfig;
      auto q_backend = static_cast<DPDKQueueConfig *>(q.common_.backend_config_);
      std::string append = "_P" + std::to_string(tx.port_id_) + "_Q" +
          std::to_string(q.common_.id_);
      auto tx_mbufs = q.common_.num_concurrent_batches_* q.common_.batch_size_;

      if (q.common_.gpu_direct_) {
        auto ext_mem = AllocateGpuPktMbuf(tx.port_id_,
                                          q.common_.max_packet_size_ - q.common_.hds_,
                                          tx_mbufs,
                                          q.common_.gpu_dev_);
        if (!ext_mem) {
          HOLOSCAN_LOG_ERROR("Failed to allocate GPU packet pool");
          return;
        }

        if (q.common_.hds_ == 0) {
          std::string gpu_name = std::string("TX_GPU_POOL") + append;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
          q_backend->pools[0] = rte_pktmbuf_pool_create_extbuf(gpu_name.c_str(), tx_mbufs,
              0, 0, ext_mem->elt_size, rte_socket_id(), std::addressof(*ext_mem), 1);
    #pragma GCC diagnostic pop
          if (q_backend->pools[0] == NULL) {
            HOLOSCAN_LOG_CRITICAL("Could not create EXT memory mempool");
            return;
          }

          HOLOSCAN_LOG_INFO("Created GPU mempool for GPUDirect: {} mbufs={} elsize={} ptr={}",
              gpu_name, tx_mbufs, ext_mem->elt_size, (void*)q_backend->pools[0]);
        } else {
          HOLOSCAN_LOG_INFO(
              "Enabling header-data split on TX with split point of {} and GPU payload size {}",
                q.common_.hds_, ext_mem->elt_size);

          std::string gpu_name = std::string("TX_GPU_POOL") + append;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
          q_backend->pools[1] = rte_pktmbuf_pool_create_extbuf(gpu_name.c_str(), tx_mbufs,
              0, 0, ext_mem->elt_size, rte_socket_id(), std::addressof(*ext_mem), 1);
    #pragma GCC diagnostic pop
          if (q_backend->pools[1] == NULL) {
            HOLOSCAN_LOG_CRITICAL("Could not create EXT memory mempool");
            return;
          }

          HOLOSCAN_LOG_INFO("Created GPU mempool for TX HDS: {} mbufs={} elsize={} ptr={}",
              gpu_name, tx_mbufs, ext_mem->elt_size, (void*)q_backend->pools[1]);

          auto cpu_name = std::string("TX_CPU_POOL") + append;
          HOLOSCAN_LOG_INFO("Creating CPU mempool for TX HDS: {} mbufs={} elsize={} ptrs={}/{}",
            cpu_name, tx_mbufs, q.common_.hds_ + RTE_PKTMBUF_HEADROOM,
            (void*)q_backend->pools[0], (void*)q_backend->pools[1]);
          q_backend->pools[0] = rte_pktmbuf_pool_create(cpu_name.c_str(),
              tx_mbufs,  MEMPOOL_CACHE_SIZE, 0, q.common_.hds_ + RTE_PKTMBUF_HEADROOM,
                  rte_socket_id());
          if (!q_backend->pools[0]) {
            HOLOSCAN_LOG_CRITICAL("Could not create sysmem mempool {} buffer split: {}",
                cpu_name, rte_errno);
            return;
          }
        }
      } else {
        auto pkt_size = q.common_.max_packet_size_ + RTE_PKTMBUF_HEADROOM;
        std::string pool_name = std::string("TX_CPU_POOL") + append;
        q_backend->pools[0] = rte_pktmbuf_pool_create(pool_name.c_str(),
            tx_mbufs, MEMPOOL_CACHE_SIZE, 0, pkt_size, rte_socket_id());
        if (q_backend->pools[0] == NULL) {
          HOLOSCAN_LOG_CRITICAL("Cannot init TX mbuf pool: {} ({})",
                        rte_errno, rte_strerror(rte_errno));
          return;
        }

        HOLOSCAN_LOG_INFO("Created CPU TX pool with packet size {} bytes and {} mbufs at {}",
              pkt_size, tx_mbufs, (void*)q_backend->pools[0]);
      }
    }

    local_port_conf[tx.port_id_].txmode.mq_mode  =  RTE_ETH_MQ_TX_NONE;
    local_port_conf[tx.port_id_].txmode.offloads =  RTE_ETH_TX_OFFLOAD_IPV4_CKSUM  |
                                                    RTE_ETH_TX_OFFLOAD_UDP_CKSUM   |
                                                    RTE_ETH_TX_OFFLOAD_TCP_CKSUM   |
                                                    RTE_ETH_TX_OFFLOAD_MULTI_SEGS;
  }

  if (SetupPoolsAndRings(max_rx_batch_size, max_tx_batch_size) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to set up pools and rings!");
    return;
  }

  for (const auto &[port, queues] : port_q_num) {
    HOLOSCAN_LOG_INFO("Initializing port {} with {} RX queues and {} TX queues...",
        port, queues.first, queues.second);

    ret = rte_eth_dev_configure(port, queues.first, queues.second, &local_port_conf[port]);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Cannot configure device: err={}, str={}, port={}",
            ret, rte_strerror(ret), port);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully configured ethdev");
    }

    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port, &default_num_rx_desc, &default_num_tx_desc);
    if (ret < 0) {
      HOLOSCAN_LOG_CRITICAL("Cannot adjust number of descriptors: err={}, port={}", ret, port);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully set descriptors");
    }

    rte_eth_macaddr_get(port, &conf_ports_eth_addr[port]);

    for (const auto &rx : cfg_.rx_) {
      if (port != rx.port_id_) {
        continue;
      }

      if (rx.flow_isolation_) {
        struct rte_flow_error error;
        ret = rte_flow_isolate(rx.port_id_, 1, &error);
        if (ret < 0) {
          HOLOSCAN_LOG_CRITICAL("Failed to set flow isolation");
        }
      }

      for (const auto &q : rx.queues_) {
        // Assume one core for now
        auto socketid = rte_lcore_to_socket_id(strtol(q.common_.cpu_cores_.c_str(), nullptr, 10));
        auto qinfo    = static_cast<DPDKQueueConfig *>(q.common_.backend_config_);

        HOLOSCAN_LOG_INFO("Setting up port:{}, queue:{}, GPUDirect:{}, Header-data split:{}",
          rx.port_id_, q.common_.id_, q.common_.gpu_direct_, q.common_.hds_);
        if (q.common_.gpu_direct_ && q.common_.hds_ > 0) {
          ret = rte_eth_rx_queue_setup(rx.port_id_, q.common_.id_,
                default_num_rx_desc, socketid, &qinfo->rxconf_qsplit, NULL);
        } else {
          ret = rte_eth_rx_queue_setup(rx.port_id_, q.common_.id_,
                default_num_rx_desc, socketid, NULL, qinfo->pools[0]);
        }

        if (ret < 0) {
          HOLOSCAN_LOG_CRITICAL("rte_eth_rx_queue_setup: err={}, port={}", ret, rx.port_id_);
          return;
        } else {
          HOLOSCAN_LOG_INFO("Successfully setup RX port {} queue {} pools: {} {}",
                rx.port_id_, q.common_.id_, (void*)qinfo->pools[0], (void*)qinfo->pools[1]);
        }
      }

      break;
    }


    struct rte_eth_txconf txq_conf;
    struct rte_eth_dev_info dev_info;

    ret = rte_eth_dev_info_get(port, &dev_info);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get device info for port {}", port);
      return;
    }

    for (const auto &tx : cfg_.tx_) {
      if (port != tx.port_id_) {
        continue;
      }

      for (const auto &q : tx.queues_) {
        txq_conf = dev_info.default_txconf;
        txq_conf.offloads = local_port_conf[tx.port_id_].txmode.offloads;
        ret = rte_eth_tx_queue_setup(tx.port_id_, q.common_.id_, default_num_tx_desc,
            rte_eth_dev_socket_id(tx.port_id_), &txq_conf);
        if (ret < 0) {
          HOLOSCAN_LOG_CRITICAL("Queue setup error {}:{}, port={} caps={:x} set={:x}",
            ret, rte_strerror(ret), tx.port_id_, dev_info.tx_offload_capa,
              local_port_conf[tx.port_id_].txmode.offloads);
          return;
        } else {
          HOLOSCAN_LOG_INFO("Successfully set up TX queue {}/{}", tx.port_id_, q.common_.id_);
        }
      }

      break;
    }
  }

  for (const auto &[port, queues] : port_q_num) {
    ret = rte_eth_dev_start(port);
    if (ret != 0) {
      HOLOSCAN_LOG_CRITICAL("Cannot start device err={}, port={}", ret, port);
      return;
    } else {
      HOLOSCAN_LOG_INFO("Successfully started port {}", port);
    }

    HOLOSCAN_LOG_INFO("Port {}, MAC address: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
      port,
      conf_ports_eth_addr[port].addr_bytes[0],
      conf_ports_eth_addr[port].addr_bytes[1],
      conf_ports_eth_addr[port].addr_bytes[2],
      conf_ports_eth_addr[port].addr_bytes[3],
      conf_ports_eth_addr[port].addr_bytes[4],
      conf_ports_eth_addr[port].addr_bytes[5]);
  }

  int flow_num = 0;
  for (const auto &rx : cfg_.rx_) {
    if (!rx.flow_isolation_) {
      rte_eth_promiscuous_enable(rx.port_id_);
    } else {
      HOLOSCAN_LOG_INFO("Not enabling promiscuous mode on port {} "
                        "since flow isolation is enabled", rx.port_id_);
    }

    for (const auto &flow : rx.flows_) {
      HOLOSCAN_LOG_INFO("Adding RX flow {}", flow.name_);
      AddFlow(rx.port_id_, flow);
    }
  }

  initialized = true;
}

int DpdkMgr::SetupPoolsAndRings(int max_rx_batch, int max_tx_batch) {
  HOLOSCAN_LOG_DEBUG("Setting up RX ring");
  rx_ring = rte_ring_create("RX_RING", 2048, rte_socket_id(),
      RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
  if (rx_ring == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
    return -1;
  }

  auto num_rx_ptrs_bufs = (1UL << 13) - 1;
  HOLOSCAN_LOG_INFO("Setting up RX burst pool with {} batches",  num_rx_ptrs_bufs);
  rx_burst_buffer = rte_mempool_create("RX_BURST_POOL",
                    num_rx_ptrs_bufs,
                    sizeof(void *) * max_rx_batch,
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

  for (const auto &tx : cfg_.tx_) {
    for (const auto &q : tx.queues_) {
      const auto append = "P" + std::to_string(tx.port_id_) + "_Q" + std::to_string(q.common_.id_);

      auto name = "TX_RING_" + append;
      HOLOSCAN_LOG_INFO("Setting up TX ring {}", name);
      uint32_t key = (tx.port_id_ << 16) | q.common_.id_;
      tx_rings[key] = rte_ring_create(name.c_str(), 2048, rte_socket_id(),
            RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
      if (tx_rings[key] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
        return -1;
      }

      name = "TX_BURST_POOL_" + append;
      tx_burst_buffers[key] = rte_mempool_create(name.c_str(),
                        (1U << 6) - 1U,
                        sizeof(void *) * max_tx_batch,
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
            name, max_tx_batch, (void*)tx_burst_buffers[key]);
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

#define MAX_PATTERN_NUM    4
#define MAX_ACTION_NUM    2

// Taken from flow_block.c DPDK example */
struct rte_flow *DpdkMgr::AddFlow(int port, const FlowConfig &cfg) {
  /* Declaring structs being used. 8< */
  struct rte_flow_attr attr;
  struct rte_flow_item pattern[MAX_PATTERN_NUM];
  struct rte_flow_action action[MAX_ACTION_NUM];
  struct rte_flow *flow = NULL;
  struct rte_flow_action_queue queue = { .index = cfg.action_.id_ };
  struct rte_flow_error error;
  struct rte_flow_item_udp udp_spec;
  struct rte_flow_item_udp udp_mask;
  struct rte_flow_item udp_item;
  /* >8 End of declaring structs being used. */
  int res;

  memset(pattern, 0, sizeof(pattern));
  memset(action, 0, sizeof(action));

  /* Set the rule attribute, only ingress packets will be checked. 8< */
  memset(&attr, 0, sizeof(struct rte_flow_attr));
  attr.ingress = 1;
  /* >8 End of setting the rule attribute. */

  /*
   * create the action sequence.
   * one action only,  move packet to queue
   */
  action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
  action[0].conf = &queue;
  action[1].type = RTE_FLOW_ACTION_TYPE_END;

  /*
   * set the first level of the pattern (ETH).
   * since in this example we just want to get the
   * ipv4 we set this level to allow all.
   */

  /* Set this level to allow all. 8< */
  pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
  pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;

  /* >8 End of setting the first level of the pattern. */
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

  attr.priority = 0;

  /* The final level must be always type end. 8< */
  pattern[3].type = RTE_FLOW_ITEM_TYPE_END;
  /* >8 End of final level must be always type end. */

  /* Validate the rule and create it. 8< */
  res = rte_flow_validate(port, &attr, pattern, action, &error);
  if (!res) {
    flow = rte_flow_create(port, &attr, pattern, action, &error);
    return flow;
  }

  return nullptr;
}

int DpdkMgr::GetRxPkts(void **pkts, int num) {
  return rte_ring_dequeue_bulk(rx_ring, pkts, num, nullptr);
}



////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void PrintDpdkStats() {
    struct rte_eth_stats eth_stats;
    int len, ret, i;
    for (i = 0; i < 2; i++) {
      rte_eth_stats_get(i, &eth_stats);
      printf("\n\nPort %u:\n", i);

      printf(" - Received packets:    %lu\n", eth_stats.ipackets);
      printf(" - Transmit packets:    %lu\n", eth_stats.opackets);
      printf(" - Received bytes:      %lu\n", eth_stats.ibytes);
      printf(" - Transmit bytes:      %lu\n", eth_stats.obytes);
      printf(" - Missed packets:      %lu\n", eth_stats.imissed);
      printf(" - Errored packets:     %lu\n", eth_stats.ierrors);
      printf(" - RX no mbufs:         %lu\n", eth_stats.rx_nombuf);

      printf("\nXStats\n");

      struct rte_eth_xstat *xstats;
      struct rte_eth_xstat_name *xstats_names;
      static const char *stats_border = "_______";

      /* Clear screen and move to top left */

      printf("PORT STATISTICS:\n================\n");
      len = rte_eth_xstats_get(i, NULL, 0);
      if (len < 0)
          rte_exit(EXIT_FAILURE,
                  "rte_eth_xstats_get(%u) failed: %d", 0,
                  len);
      xstats = (struct rte_eth_xstat *)calloc(len, sizeof(*xstats));
      if (xstats == NULL)
          rte_exit(EXIT_FAILURE,
                  "Failed to calloc memory for xstats");
      ret = rte_eth_xstats_get(i, xstats, len);
      if (ret < 0 || ret > len) {
          free(xstats);
          rte_exit(EXIT_FAILURE,
                  "rte_eth_xstats_get(%u) len%i failed: %d",
                  0, len, ret);
      }
      xstats_names = (struct rte_eth_xstat_name *)calloc(len, sizeof(*xstats_names));
      if (xstats_names == NULL) {
          free(xstats);
          rte_exit(EXIT_FAILURE,
                  "Failed to calloc memory for xstats_names");
      }
      ret = rte_eth_xstats_get_names(i, xstats_names, len);
      if (ret < 0 || ret > len) {
          free(xstats);
          free(xstats_names);
          rte_exit(EXIT_FAILURE,
                  "rte_eth_xstats_get_names(%u) len%i failed: %d",
                  0, len, ret);
      }
      for (i = 0; i < len; i++) {
          if (xstats[i].value > 0)
              printf("Port %u: %s %s:\t\t%lu\n",
                      0, stats_border,
                      xstats_names[i].name,
                      xstats[i].value);
      }
      fflush(stdout);
      free(xstats);
      free(xstats_names);
  }
}

DpdkMgr::~DpdkMgr() {
  PrintDpdkStats();
}


////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::Run() {
  int secondary_id = 0;
  int icore;

  HOLOSCAN_LOG_INFO("Starting advanced network workers");
  // determine the correct process types for input/output
  int (*rx_worker)(void*) = rx_core;
  int (*tx_worker)(void*) = tx_core;

  for (auto &rx : cfg_.rx_) {
    if (rx.empty) {
      continue;
    }
    for (auto &q : rx.queues_) {
      auto qinfo = static_cast<DPDKQueueConfig *>(q.common_.backend_config_);
      auto params = new RxWorkerParams;
      params->gpu_direct = q.common_.gpu_direct_;
      params->hds    = q.common_.hds_ > 0;
      params->port   = rx.port_id_;
      params->ring   = rx_ring;
      params->queue  = q.common_.id_;
      params->burst_pool   = rx_burst_buffer;
      params->meta_pool  = rx_meta;
      params->batch_size = q.common_.batch_size_;
      rte_eal_remote_launch(rx_worker, (void*)params,
          strtol(q.common_.cpu_cores_.c_str(), NULL, 10));
    }
  }

  for (auto &tx : cfg_.tx_) {
    if (tx.empty) {
      continue;
    }
    for (auto &q : tx.queues_) {
      uint32_t key = (tx.port_id_ << 16) | q.common_.id_;
      auto qinfo = static_cast<DPDKQueueConfig *>(q.common_.backend_config_);
      auto params = new TxWorkerParams;
      //  params->hds    = q.common_.hds_ > 0;
      params->port   = tx.port_id_;
      params->ring   = tx_rings[key];
      params->queue  = q.common_.id_;
      params->burst_pool  = tx_burst_buffers[key];
      params->meta_pool   = tx_meta;
      params->batch_size  = q.common_.batch_size_;
      rte_eth_macaddr_get(tx.port_id_, &params->mac_addr);
      rte_eal_remote_launch(tx_worker, (void*)params,
          strtol(q.common_.cpu_cores_.c_str(), NULL, 10));
    }
  }

  HOLOSCAN_LOG_INFO("Done starting workers");
}

void DpdkMgr::wait() {
  int icore = 0;

  HOLOSCAN_LOG_INFO("Stopping all workers");
  force_quit.store(true);
  RTE_LCORE_FOREACH_WORKER(icore) {
    if (rte_eal_wait_lcore(icore) < 0) {
      fprintf(stderr, "bad exit for coreid: %d\n", icore);
      break;
    }
  }

  PrintDpdkStats();
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
void DpdkMgr::flush_packets(int port) {
  struct rte_mbuf * rx_mbuf;
  HOLOSCAN_LOG_INFO("Flushing packet on port {}", port);
  while (rte_eth_rx_burst(port, 0, &rx_mbuf, 1) != 0) {
    rte_pktmbuf_free(rx_mbuf);
  }
}

void DpdkMgr::check_pkts_to_free(rte_ring *msg_ring,
    rte_mempool *burst_pool, rte_mempool *meta_pool) {
    // AdvNetBurstParams *msg;
    // if (rte_ring_dequeue(msg_ring, reinterpret_cast<void**>(&msg)) == 0) {
    //   for (int p = 0; p < msg->hdr.hdr.num_pkts; p++) {
    //     rte_pktmbuf_free_seg(reinterpret_cast<rte_mbuf*>(msg->cpu_pkts[p]));
    //   }
    //   rte_mempool_put(burst_pool, msg->cpu_pkts);
    //   rte_mempool_put(meta_pool, msg);
    // }
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////
int DpdkMgr::rx_core(void *arg) {
  RxWorkerParams *tparams = (RxWorkerParams*)arg;
  struct rte_mbuf * rx_mbufs[DEFAULT_NUM_RX_BURST];
  int ret = 0;
  uint64_t freq = rte_get_tsc_hz();
  uint64_t timeout_ticks = freq * 0.02;  // expect all packets within 20ms

  uint64_t total_pkts = 0;

  flush_packets(tparams->port);
  struct rte_mbuf* mbuf_arr[DEFAULT_NUM_RX_BURST];

  HOLOSCAN_LOG_INFO("Starting RX Core {}, port {}, queue {}, socket {}",
      rte_lcore_id(), tparams->port, tparams->queue, rte_socket_id());
  int nb_rx = 0;
  int to_copy = 0;
  //
  //  run loop
  //
  while (!force_quit.load()) {
    AdvNetBurstParams *burst;
    if (rte_mempool_get(tparams->meta_pool, reinterpret_cast<void **>(&burst)) < 0) {
      HOLOSCAN_LOG_ERROR("Processing function falling behind. No free buffers for metadata!");
      exit(1);
    }

    //  Queue ID for receiver to differentiate
    burst->hdr.hdr.q_id = tparams->queue;
    burst->hdr.hdr.port_id = tparams->port;

    if (!tparams->gpu_direct || tparams->hds) {
      if (rte_mempool_get(tparams->burst_pool, reinterpret_cast<void **>(&burst->cpu_pkts)) < 0) {
        HOLOSCAN_LOG_ERROR("Processing function falling behind. No free CPU buffers for packets!");
        continue;
      }
    } else {
      burst->cpu_pkts = nullptr;
    }

    if (tparams->gpu_direct) {
      if (rte_mempool_get(tparams->burst_pool, reinterpret_cast<void **>(&burst->gpu_pkts)) < 0) {
        HOLOSCAN_LOG_ERROR("Processing function falling behind. No free GPU buffers for packets!");
        if (!tparams->gpu_direct || tparams->hds) {
          rte_mempool_put(tparams->burst_pool, burst->cpu_pkts);
        }
        continue;
      }
    } else {
      burst->gpu_pkts = nullptr;
    }

    if (nb_rx > 0) {
      burst->hdr.hdr.num_pkts = nb_rx;

      if (!tparams->gpu_direct || tparams->hds) {
        memcpy(&burst->cpu_pkts[0], &mbuf_arr[to_copy], sizeof(rte_mbuf*) * nb_rx);

        if (tparams->hds) {
          for (int p = 0; p < nb_rx; p++) {
            burst->gpu_pkts[p] = mbuf_arr[to_copy + p]->next;
          }
        }
      } else {
        memcpy(&burst->gpu_pkts[0], &mbuf_arr[to_copy], sizeof(rte_mbuf*) * nb_rx);
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

      nb_rx  = rte_eth_rx_burst(tparams->port, tparams->queue,
          reinterpret_cast<rte_mbuf**>(&mbuf_arr[0]), DEFAULT_NUM_RX_BURST);

      if (nb_rx == 0) {
        continue;
      }

      to_copy       = std::min(nb_rx, (int)(tparams->batch_size - burst->hdr.hdr.num_pkts));

      if (!tparams->gpu_direct || tparams->hds) {
        memcpy(&burst->cpu_pkts[burst->hdr.hdr.num_pkts], mbuf_arr, sizeof(rte_mbuf*) * to_copy);
        if (tparams->hds) {
          for (int p = 0; p < to_copy; p++) {
            burst->gpu_pkts[burst->hdr.hdr.num_pkts + p] = mbuf_arr[p]->next;
          }
        }
      } else {
        memcpy(&burst->gpu_pkts[burst->hdr.hdr.num_pkts], mbuf_arr, sizeof(rte_mbuf*) * to_copy);
      }

      burst->hdr.hdr.num_pkts += to_copy;
      total_pkts          += nb_rx;
      nb_rx               -= to_copy;

      if (burst->hdr.hdr.num_pkts == tparams->batch_size) {
        rte_ring_enqueue(tparams->ring, reinterpret_cast<void *>(burst));
        break;
      }
    } while (!force_quit.load());
  }

  HOLOSCAN_LOG_ERROR("Total packets received by application (port/queue {}/{}): {}\n",
        tparams->port, tparams->queue, total_pkts);
  return 0;
}



int DpdkMgr::tx_core(void *arg) {
  TxWorkerParams *tparams = (TxWorkerParams*)arg;
  uint64_t seq;
  uint64_t pkts_tx = 0;
  AdvNetBurstParams *msg;
  int64_t bursts = 0;

  HOLOSCAN_LOG_INFO("Starting TX Core {}, port {}, queue {} socket {} using burst pool {} ring {}",
        rte_lcore_id(), tparams->port, tparams->queue, rte_socket_id(),
        (void*)tparams->burst_pool, (void*)tparams->ring);

  while (!force_quit.load()) {
    if (rte_ring_dequeue(tparams->ring, reinterpret_cast<void**>(&msg)) != 0) {
      continue;
    }

    // Header-data split needs to chain all the buffers
    if (msg->cpu_pkts != nullptr && msg->gpu_pkts != nullptr) {
      for (size_t p = 0; p < msg->hdr.hdr.num_pkts; p++) {
        auto *cpu_mbuf = reinterpret_cast<rte_mbuf*>(msg->cpu_pkts[p]);
        auto *gpu_mbuf = reinterpret_cast<rte_mbuf*>(msg->gpu_pkts[p]);

        cpu_mbuf->next = gpu_mbuf;
        gpu_mbuf->next = nullptr;

        cpu_mbuf->nb_segs = 2;
      }
    }

    HOLOSCAN_LOG_DEBUG("Got burst in TX");

    if (msg->cpu_pkts != nullptr) {
      for (size_t p = 0; p < msg->hdr.hdr.num_pkts; p++) {
        auto *mbuf = reinterpret_cast<rte_mbuf*>(msg->cpu_pkts[p]);
        auto *pkt  = rte_pktmbuf_mtod(mbuf, uint8_t*);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
        rte_ether_addr_copy(&tparams->mac_addr, reinterpret_cast<rte_ether_addr *>(pkt + 6));
#pragma GCC diagnostic pop
      }
    }

    auto pkts_to_transmit = static_cast<int64_t>(msg->hdr.hdr.num_pkts);

    size_t pkts_tx = 0;
    while (pkts_tx != msg->hdr.hdr.num_pkts && !force_quit.load()) {
      auto to_send = static_cast<uint16_t>(
            std::min(static_cast<size_t>(DEFAULT_NUM_TX_BURST), msg->hdr.hdr.num_pkts - pkts_tx));

      // CPU-only or HDS mode
      int tx;
      if (msg->cpu_pkts != nullptr) {
        tx = rte_eth_tx_burst(tparams->port,
              tparams->queue, reinterpret_cast<rte_mbuf**>(&msg->cpu_pkts[pkts_tx]), to_send);
      } else {
        tx = rte_eth_tx_burst(tparams->port,
              tparams->queue, reinterpret_cast<rte_mbuf**>(&msg->gpu_pkts[pkts_tx]), to_send);
      }

      pkts_tx += tx;
    }

    if (msg->cpu_pkts) {
      rte_mempool_put(tparams->burst_pool, static_cast<void*>(msg->cpu_pkts));
    }

    if (msg->gpu_pkts) {
      rte_mempool_put(tparams->burst_pool, static_cast<void*>(msg->gpu_pkts));
    }

    rte_mempool_put(tparams->meta_pool, msg);

    HOLOSCAN_LOG_DEBUG("Sent {} packets\n", pkts_tx);

    bursts++;
  }
  printf("TX thread exiting\n");

  return 0;
}
};  // namespace holoscan::ops
