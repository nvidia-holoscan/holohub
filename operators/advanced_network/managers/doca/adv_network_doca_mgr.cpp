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
#include <unistd.h>
#include "adv_network_doca_mgr.h"
#include "adv_network_doca_kernels.h"
#include "holoscan/holoscan.hpp"
// #include <cudaProfiler.h>

using namespace std::chrono;

namespace holoscan::ops {

DocaMgr doca_mgr{};

std::atomic<bool> force_quit_doca = false;
uint64_t stats_rx_tot_pkts;
uint64_t stats_rx_tot_bytes;
uint64_t stats_rx_tot_batch;

uint64_t stats_tx_tot_pkts;
uint64_t stats_tx_tot_bytes;
uint64_t stats_tx_tot_batch;

struct TxDocaWorkerQueue {
  int queue;
  int gpu_id;
  uint64_t tx_pkts = 0;
  uint32_t batch_size;
  struct rte_ring* ring;
  DocaTxQueue* txq;
};

struct TxDocaWorkerParams {
  int port;
  int txqn;
  uint32_t batch_size;
  struct rte_mempool* meta_pool;
  struct rte_ether_addr mac_addr;
  struct doca_gpu* gdev;
  struct TxDocaWorkerQueue txqw[MAX_NUM_TX_QUEUES];
};

struct RxDocaWorkerQueue {
  int queue;
  int gpu_id;
  uint64_t rx_pkts = 0;
  uint32_t batch_size;
  DocaRxQueue* rxq;
};

struct RxDocaWorkerParams {
  int port;
  int rxqn;
  struct rte_ring* ring;
  struct rte_mempool* meta_pool;
  struct doca_gpu* gdev;
  struct RxDocaWorkerQueue rxqw[MAX_NUM_RX_QUEUES];
};

////////////////////////////////////////////////////////////////////////////////
///
///  \brief Init
///
////////////////////////////////////////////////////////////////////////////////

struct rte_eth_dev_info dev_info = {0};
struct rte_eth_conf eth_conf = {0};
struct rte_mempool* mp;

/*
 * DOCA PE callback to be invoked on Eth Txq to get the debug info
 * when sending packets and decrease number of posted completions.
 *
 * @event_notify [in]: DOCA PE event debug handler
 * @event_user_data [in]: custom user data set at registration time
 */
void decrease_txq_completion_cb(struct doca_eth_txq_gpu_event_notify_send_packet* event_notify,
                                union doca_data event_user_data) {
  ((std::atomic<uint32_t>*)event_user_data.u64)[0]--;
  HOLOSCAN_LOG_DEBUG("Queue cmp {}", ((std::atomic<uint32_t>*)event_user_data.u64)[0].load());
}

/*
 * Initialize a DOCA device with PCIe address object.
 *
 * @pcie_value [in]: PCIe address object
 * @retval [out]: DOCA device object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t open_doca_device_with_pci(const char* pcie_value, struct doca_dev** retval) {
  struct doca_devinfo** dev_list;
  uint32_t nb_devs;
  doca_error_t res;
  size_t i;
  uint8_t is_addr_equal = 0;

  /* Set default return value */
  *retval = nullptr;

  res = doca_devinfo_create_list(&dev_list, &nb_devs);
  if (res != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to load doca devices list. Doca_error value: %d", res);
    return res;
  }

  /* Search */
  for (i = 0; i < nb_devs; i++) {
    res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
    if (res == DOCA_SUCCESS && is_addr_equal) {
      /* if device can be opened */
      res = doca_dev_open(dev_list[i], retval);
      if (res == DOCA_SUCCESS) {
        doca_devinfo_destroy_list(dev_list);
        return res;
      }
    }
  }

  HOLOSCAN_LOG_CRITICAL("Matching device not found");
  res = DOCA_ERROR_NOT_FOUND;

  doca_devinfo_destroy_list(dev_list);
  return res;
}

static doca_error_t init_doca_device(char** argv, int argc, const char* nic_pcie_addr,
                                     struct doca_dev** ddev, uint16_t* dpdk_port_id) {
  doca_error_t result;
  int ret;
  // char *eal_param[3] = {"", "-a", "00:00.0"};

  if (nic_pcie_addr == nullptr || ddev == nullptr || dpdk_port_id == nullptr)
    return DOCA_ERROR_INVALID_VALUE;

  if (strlen(nic_pcie_addr) >= DOCA_DEVINFO_PCI_ADDR_SIZE) return DOCA_ERROR_INVALID_VALUE;

  result = open_doca_device_with_pci(nic_pcie_addr, ddev);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to open NIC device based on PCI address");
    return result;
  }

  ret = rte_eal_init(argc, argv);
  if (ret < 0) {
    HOLOSCAN_LOG_CRITICAL("DPDK init failed: %d", ret);
    return DOCA_ERROR_DRIVER;
  }

  /* Enable DOCA Flow HWS mode */
  result = doca_dpdk_port_probe(*ddev, "dv_flow_en=2");
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Function doca_dpdk_port_probe returned %s",
                          doca_error_get_descr(result));
    return result;
  }

  result = doca_dpdk_get_first_port_id(*ddev, dpdk_port_id);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Function doca_dpdk_get_first_port_id returned %s",
                          doca_error_get_descr(result));
    return result;
  }

#if 0
    struct doca_log_backend *stdout_logger = nullptr;

    result = doca_log_backend_create_with_file_sdk(stdout, &stdout_logger);
    if (result != DOCA_SUCCESS)
            return result;

    result = doca_log_backend_set_sdk_level(stdout_logger, DOCA_LOG_LEVEL_TRACE);
    if (result != DOCA_SUCCESS)
            return result;
#endif

  return DOCA_SUCCESS;
}

struct doca_flow_port* DocaMgr::init_doca_flow(uint16_t port_id, uint8_t rxq_num) {
  doca_error_t result;
  char port_id_str[MAX_PORT_STR_LEN];
  struct doca_flow_port_cfg* port_cfg;
  struct doca_flow_port* df_port;
  struct doca_flow_cfg* rxq_flow_cfg;
  doca_error_t ret = DOCA_SUCCESS;
  int ret_dpdk = 0;
  struct rte_eth_dev_info dev_info = {0};
  struct rte_eth_conf eth_conf = {
      .rxmode = {
              .mtu = 2048, /* Not really used, just to initialize DPDK */
          },
  };
  struct rte_mempool* mp = nullptr;
  struct rte_flow_error error;

  /*
   * DPDK should be initialized and started before DOCA Flow.
   * DPDK doesn't start the device without, at least, one DPDK Rx queue.
   * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
   *
   * Following lines of code can be considered the minimum WAR for this issue.
   */

  ret_dpdk = rte_eth_dev_info_get(port_id, &dev_info);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
    return nullptr;
  }

  ret_dpdk = rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
    return nullptr;
  }

  mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, 8192, rte_eth_dev_socket_id(port_id));
  if (mp == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
    return nullptr;
  }

  for (int idx = 0; idx < rxq_num; idx++) {
    ret_dpdk =
        rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), nullptr, mp);
    if (ret) {
      HOLOSCAN_LOG_CRITICAL("Failed rte_eth_rx_queue_setup with: %s", rte_strerror(-ret));
      return nullptr;
    }
  }

  ret_dpdk = rte_flow_isolate(port_id, 1, &error);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_flow_isolate with: %s", error.message);
    return nullptr;
  }

  ret_dpdk = rte_eth_dev_start(port_id);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
    return nullptr;
  }

  /* Initialize doca flow framework */
  ret = doca_flow_cfg_create(&rxq_flow_cfg);
  if (ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to create doca_flow_cfg: %s", doca_error_get_descr(ret));
    return nullptr;
  }

  ret = doca_flow_cfg_set_pipe_queues(rxq_flow_cfg, rxq_num);
  if (ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg pipe_queues: %s", doca_error_get_descr(ret));
    doca_flow_cfg_destroy(rxq_flow_cfg);
    return nullptr;
  }

  /*
   * HWS: Hardware steering
   * Isolated: don't create RSS rule for DPDK created RX queues
   */
  ret = doca_flow_cfg_set_mode_args(rxq_flow_cfg, "vnf,hws,isolated");
  if (ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg mode_args: %s", doca_error_get_descr(ret));
    doca_flow_cfg_destroy(rxq_flow_cfg);
    return nullptr;
  }

  ret = doca_flow_cfg_set_nr_counters(rxq_flow_cfg, FLOW_NB_COUNTERS);
  if (ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg nr_counters: %s", doca_error_get_descr(ret));
    doca_flow_cfg_destroy(rxq_flow_cfg);
    return nullptr;
  }

  result = doca_flow_init(rxq_flow_cfg);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to init doca flow with: %s", doca_error_get_descr(result));
    return nullptr;
  }

  doca_flow_cfg_destroy(rxq_flow_cfg);

  /* Start doca flow port */
  result = doca_flow_port_cfg_create(&port_cfg);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to create doca_flow_port_cfg: %s", doca_error_get_descr(result));
    return nullptr;
  }
  snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_id);
  result = doca_flow_port_cfg_set_devargs(port_cfg, port_id_str);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_port_cfg devargs: %s",
                          doca_error_get_descr(result));
    doca_flow_port_cfg_destroy(port_cfg);
    return nullptr;
  }
  result = doca_flow_port_start(port_cfg, &df_port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to start doca flow port with: %s", doca_error_get_descr(result));
    doca_flow_port_cfg_destroy(port_cfg);
    return nullptr;
  }
  doca_flow_port_cfg_destroy(port_cfg);

  return df_port;
}

int DocaMgr::setup_pools_and_rings(int max_rx_batch, int max_tx_batch) {
  AdvNetBurstParams* bursts_rx[(1U << 6) - 1U];
  AdvNetBurstParams* bursts_tx[(1U << 7) - 1U];
  int idx = 0;

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

  while (idx < (1U << 6) - 1U &&
         rte_mempool_get(rx_meta, reinterpret_cast<void**>(&bursts_rx[idx])) == 0) {
    bursts_rx[idx]->gpu_pkts = (void**)calloc(CUDA_MAX_RX_NUM_PKTS, sizeof(void*));
    bursts_rx[idx]->cpu_pkts = nullptr;
    idx++;
  }

  rte_mempool_put_bulk(rx_meta, reinterpret_cast<void**>(&bursts_rx), idx);

  for (const auto& tx : cfg_.tx_) {
    for (const auto& q : tx.queues_) {
      const auto append = "P" + std::to_string(tx.port_id_) + "_Q" + std::to_string(q.common_.id_);
      auto name = "TX_RING_" + append;
      HOLOSCAN_LOG_INFO("Setting up TX ring {}", name);
      uint32_t key = (tx.port_id_ << 16) | q.common_.id_;
      tx_rings[key] = rte_ring_create(
          name.c_str(), 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
      if (tx_rings[key] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
        return -1;
      }
    }
  }

  HOLOSCAN_LOG_INFO("Setting up TX meta pool");
  tx_meta = rte_mempool_create("TX_META_POOL",
                               (1U << 7) - 1U,
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

  idx = 0;
  while (idx < (1U << 7) - 1U &&
         rte_mempool_get(tx_meta, reinterpret_cast<void**>(&bursts_tx[idx])) == 0) {
    bursts_tx[idx]->pkts[0] = (void**)calloc(1, sizeof(void*));
    cudaMallocHost(&bursts_tx[idx]->pkt_lens[0], max_tx_batch * sizeof(uint32_t));
    memset(bursts_tx[idx]->pkt_lens[0], 0, max_tx_batch * sizeof(uint32_t));
    idx++;
  }

  rte_mempool_put_bulk(tx_meta, reinterpret_cast<void**>(&bursts_tx), idx);

  return 0;
}

void DocaMgr::set_config_and_initialize(const AdvNetConfigYaml& cfg) {
  if (!this->initialized_) {
    cfg_ = cfg;
    cpu_set_t mask;
    long nproc, i;
    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&DocaMgr::initialize, this);
    t.join();

    this->initialized_ = true;
    run();
  }
}

void DocaMgr::initialize() {
  int ret;
  doca_error_t doca_ret;
  /* Initialize DPDK params */
  constexpr int max_nargs = 32;
  constexpr int max_arg_size = 64;
  int max_tx_batch_size = 0;
  int max_packet_size = 0;
  char** argv_;
  int arg = 0;
  argv_ = (char**)malloc(sizeof(char*) * max_nargs);
  for (int i = 0; i < max_nargs; i++) { argv_[i] = (char*)malloc(max_arg_size); }

  std::string cores = std::to_string(cfg_.common_.master_core_) + ",";  // Master core must be first

  for (const auto& rx : cfg_.rx_) {
    for (const auto& q : rx.queues_) cores += q.common_.cpu_cores_ + ",";
  }

  for (const auto& tx : cfg_.tx_) {
    for (const auto& q : tx.queues_) cores += q.common_.cpu_cores_ + ",";
  }

  cores = cores.substr(0, cores.size() - 1);
  std::cout << cores;

  strncpy(argv_[arg++], "-l", max_arg_size - 1);
  strncpy(argv_[arg++], cores.c_str(), max_arg_size - 1);
  strncpy(argv_[arg++], "-a", max_arg_size - 1);
  strncpy(argv_[arg++], std::string("00:00.0").c_str(), max_arg_size - 1);

  rxq_num = 0;
  txq_num = 0;

  /* All RX queues must be on the same port. */
  if (cfg_.rx_.size() > 0) {
    rxq_num = cfg_.rx_[0].queues_.size();

    doca_ret =
        init_doca_device(argv_, arg, cfg_.rx_[0].if_name_.c_str(), &ddev_rx, &cfg_.rx_[0].port_id_);
    if (doca_ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed init DOCA device {}", net_bdf);
      return;
    }

    for (auto& rx : cfg_.rx_) rx.port_id_ = cfg_.rx_[0].port_id_;

    if (cudaDeviceGetPCIBusId(gpu_bdf, sizeof(gpu_bdf), cfg_.rx_[0].queues_[0].common_.gpu_dev_) !=
        cudaSuccess) {
      HOLOSCAN_LOG_CRITICAL("Failed get GPU PCIe addr device {}",
                            cfg_.rx_[0].queues_[0].common_.gpu_dev_);
      return;
    }
  }

  /* All TX queues must be on the same port. */
  if (cfg_.tx_.size() > 0) {
    txq_num = cfg_.tx_[0].queues_.size();
    ddev_tx = ddev_rx;

    if ((cfg_.rx_.size() == 0) ||
        (cfg_.rx_.size() > 0 && cfg_.rx_[0].if_name_.compare(cfg_.tx_[0].if_name_) != 0)) {
      doca_ret = init_doca_device(
          argv_, arg, cfg_.tx_[0].if_name_.c_str(), &ddev_tx, &cfg_.tx_[0].port_id_);
      if (doca_ret != DOCA_SUCCESS) {
        HOLOSCAN_LOG_CRITICAL("Failed init DOCA device {}", net_bdf);
        return;
      }

      for (auto& tx : cfg_.tx_) tx.port_id_ = cfg_.tx_[0].port_id_;
    } else {
      for (auto& tx : cfg_.tx_) tx.port_id_ = cfg_.rx_[0].port_id_;
    }

    // Need to create 2 GPU devices in case of RX + TX support!!!
    if (cudaDeviceGetPCIBusId(gpu_bdf, sizeof(gpu_bdf), cfg_.tx_[0].queues_[0].common_.gpu_dev_) !=
        cudaSuccess) {
      HOLOSCAN_LOG_CRITICAL("Failed get GPU PCIe addr device {}",
                            cfg_.tx_[0].queues_[0].common_.gpu_dev_);
      return;
    }
  }

  doca_ret = doca_gpu_create(gpu_bdf, &gdev);
  if (doca_ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed get DOCA GPU device {}", gpu_bdf);
    return;
  }

  rte_eth_macaddr_get(dpdk_port_id, &mac_addr);

  HOLOSCAN_LOG_INFO("DOCA init -- RX: {} TX: {}",
                    cfg_.rx_.size() > 0 ? "ENABLED" : "DISABLED",
                    cfg_.tx_.size() > 0 ? "ENABLED" : "DISABLED");

  // For now make a single queue. Support more sophisticated TX on next release
  for (auto& tx : cfg_.tx_) {
    for (auto& q : tx.queues_) {
      max_tx_batch_size = std::max(max_tx_batch_size, q.common_.batch_size_);
      max_packet_size = std::max(max_packet_size, q.common_.max_packet_size_);
    }
  }

  if (setup_pools_and_rings(0, max_tx_batch_size) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to set up pools and rings!");
    return;
  }

  if (rxq_num > 0) {
    df_port = init_doca_flow(dpdk_port_id, rxq_num);
    if (df_port == nullptr) {
      HOLOSCAN_LOG_CRITICAL("FAILED: init_doca_flow");
      return;
    }
  }

  // Rx queues single port allowed
  for (auto& rx : cfg_.rx_) {
    for (auto& q : rx.queues_) {
      HOLOSCAN_LOG_INFO(
          "Configuring RX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, rx.port_id_);
      rxq_pkts = q.common_.num_concurrent_batches_ * q.common_.batch_size_;

      if (!rte_is_power_of_2(rxq_pkts)) rxq_pkts = rte_align32pow2(rxq_pkts);

      q.common_.backend_config_ = new DocaRxQueue(
          ddev_rx, gdev, df_port, q.common_.id_, rxq_pkts, q.common_.max_packet_size_);
    }
  }

  // Tx queues single port allowed
  for (auto& tx : cfg_.tx_) {
    for (auto& q : tx.queues_) {
      HOLOSCAN_LOG_INFO(
          "Configuring TX queue: {} ({}) on port {}", q.common_.name_, q.common_.id_, tx.port_id_);

      txq_pkts = next_power_of_two(q.common_.num_concurrent_batches_) *
                 next_power_of_two((uint64_t)q.common_.batch_size_);
      q.common_.backend_config_ = new DocaTxQueue(ddev_tx,
                                                  gdev,
                                                  q.common_.id_,
                                                  txq_pkts,
                                                  q.common_.max_packet_size_,
                                                  &decrease_txq_completion_cb);
    }
  }

  if (cfg_.rx_.size() > 0) {
    create_default_pipe(cfg_.rx_[0].queues_.size() - cfg_.rx_[0].flows_.size());

    int flow_num = 0;
    for (auto& rx : cfg_.rx_) {
      for (auto& flow : rx.flows_) {
        HOLOSCAN_LOG_INFO("Create RX flow {} to queue {}", flow.name_, flow.action_.id_);
        for (auto& q : rx.queues_) {
          auto q_backend = static_cast<DocaRxQueue*>(q.common_.backend_config_);
          if (q_backend->qid == flow.action_.id_) {
            q_backend->create_udp_pipe(flow, rxq_pipe_default);
            flow.backend_config_ = q.common_.backend_config_;
          }
        }
      }
    }

    if (rxq_num > 0) {
      doca_ret = create_root_pipe();
      if (doca_ret != DOCA_SUCCESS) HOLOSCAN_LOG_CRITICAL("Can't create UDP root pipe");
    }

    /* Create semaphore for GPU - CPU communication per rxq*/
    for (auto& rx : cfg_.rx_) {
      for (auto& q : rx.queues_) {
        HOLOSCAN_LOG_INFO("Create RX semaphore");
        auto q_backend = static_cast<DocaRxQueue*>(q.common_.backend_config_);
        q_backend->create_semaphore();
      }
    }
  }

  /* Tx burst preallocate */
  HOLOSCAN_LOG_INFO("max_tx_batch_size {} max_packet_size {}", max_tx_batch_size, max_packet_size);
  for (int idx = 0; idx < MAX_TX_BURST; idx++) {
    cudaMallocHost(&(burst[idx].gpu_pkts_len), max_tx_batch_size * sizeof(uint32_t));
    burst[idx].hdr.hdr.max_pkt_size = max_packet_size;
  }

  rxq_pipe_default = nullptr;
  initialized = true;
  stats_rx_tot_pkts = 0;
  stats_rx_tot_bytes = 0;
  stats_rx_tot_batch = 0;

  stats_tx_tot_pkts = 0;
  stats_tx_tot_bytes = 0;
  stats_tx_tot_batch = 0;
  burst_tx_idx = 0;
  // cudaProfilerStart();
}

doca_error_t DocaMgr::create_default_pipe(uint32_t cnt_defq) {
  uint16_t flow_queue_id;
  uint16_t rss_queues[MAX_DEFAULT_QUEUES];
  int idxq = 0;
  doca_error_t result;
  struct doca_flow_match match = {0};
  struct doca_flow_match match_mask = {0};
  struct doca_flow_fwd fwd = {};
  struct doca_flow_fwd miss_fwd = {};
  struct doca_flow_pipe_cfg* pipe_cfg;
  struct doca_flow_pipe_entry* entry;
  struct doca_flow_monitor monitor = {
      .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
  };
  const char* pipe_name = "GPU_RXQ_UDP_DEF_PIPE";

  if (cnt_defq >= MAX_DEFAULT_QUEUES) {
    HOLOSCAN_LOG_CRITICAL("Too many default queues {}", cnt_defq);
    return DOCA_ERROR_INVALID_VALUE;
  }

  if (cnt_defq == 0) {
    HOLOSCAN_LOG_CRITICAL("No need for a default queue");
    return DOCA_ERROR_INVALID_VALUE;
  }

  match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
  // match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

  result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg name: %s", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg enable_strict_matching: %s",
                       doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg type: %s", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, false);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg is_root: %s",
                       doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, &match_mask);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg monitor: %s",
                       doca_error_get_descr(result));
    return result;
  }

  bool create_pipe;
  for (const auto& rx : cfg_.rx_) {
    for (auto& q : rx.queues_) {
      create_pipe = true;
      auto q_backend = static_cast<DocaRxQueue*>(q.common_.backend_config_);

      for (auto& rx2 : cfg_.rx_) {
        for (auto& flow : rx2.flows_) {
          if (q_backend->qid == flow.action_.id_) create_pipe = false;
        }
      }

      if (create_pipe == true) {
        // Add default entries
        doca_eth_rxq_get_flow_queue_id(q_backend->eth_rxq_cpu, &flow_queue_id);
        rss_queues[idxq] = flow_queue_id;
        HOLOSCAN_LOG_DEBUG("create_default_pipe idx {} queue {}", idxq, flow_queue_id);
        idxq++;
      }
    }
  }

  fwd.type = DOCA_FLOW_FWD_RSS;
  fwd.rss_queues = rss_queues;
  fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4;
  fwd.num_of_queues = cnt_defq;

  miss_fwd.type = DOCA_FLOW_FWD_DROP;

  result = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, &(rxq_pipe_default));
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe creation failed with: {}", doca_error_get_descr(result));
    return result;
  }

  /* Add HW offload */
  result = doca_flow_pipe_add_entry(
      0, rxq_pipe_default, &match, nullptr, nullptr, nullptr, DOCA_FLOW_NO_WAIT, nullptr, &entry);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe entry creation failed with: {}", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_entries_process(df_port, 0, default_flow_timeout_usec, 0);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe entry process failed with: {}", doca_error_get_descr(result));
    return result;
  }

  HOLOSCAN_LOG_INFO("Created Default Pipe {}", pipe_name);

  return DOCA_SUCCESS;
}

doca_error_t DocaMgr::create_root_pipe() {
  doca_error_t result;
  uint32_t cnt_defq = cfg_.rx_[0].queues_.size() - cfg_.rx_[0].flows_.size();

  struct doca_flow_match match_mask = {0};
  struct doca_flow_match udp_match = {0};
  struct doca_flow_monitor monitor = {
      .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
  };
  struct doca_flow_pipe_cfg* pipe_cfg;
  const char* pipe_name = "ROOT_PIPE";
  memset(&match_mask, 0, sizeof(match_mask));

  result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg name: %s", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg enable_strict_matching: %s",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_CONTROL);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg type: %s", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg is_root: %s",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_match(pipe_cfg, nullptr, &match_mask);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg monitor: %s",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }

  result = doca_flow_pipe_create(pipe_cfg, nullptr, nullptr, &root_pipe);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Root pipe creation failed with: %s", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  doca_flow_pipe_cfg_destroy(pipe_cfg);

  udp_match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
  udp_match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

  for (const auto& rx : cfg_.rx_) {
    for (const auto& flow : rx.flows_) {
      HOLOSCAN_LOG_INFO("Adding RX flow {} from {} to control pipe", flow.name_, flow.action_.id_);
      auto q_backend = static_cast<DocaRxQueue*>(flow.backend_config_);

      struct doca_flow_fwd udp_fwd = {
          .type = DOCA_FLOW_FWD_PIPE,
          .next_pipe = q_backend->rxq_pipe,
      };

      // Reqork priority if you have multiple queues!!!
      result = doca_flow_pipe_control_add_entry(0,
                                                0,
                                                root_pipe,
                                                &udp_match,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                &udp_fwd,
                                                nullptr,
                                                &(q_backend->root_udp_entry));

      if (result != DOCA_SUCCESS) {
        HOLOSCAN_LOG_CRITICAL("Root pipe UDP entry creation failed with: {}",
                              doca_error_get_descr(result));
        return result;
      }
    }
  }

  if (cnt_defq > 0) {
    // Add default pipe
    struct doca_flow_fwd udp_fwd = {
        .type = DOCA_FLOW_FWD_PIPE,
        .next_pipe = rxq_pipe_default,
    };

    // Lower priority than UDP + port filters
    HOLOSCAN_LOG_INFO("Adding RX default pipeline");
    result = doca_flow_pipe_control_add_entry(0,
                                              1,
                                              root_pipe,
                                              &udp_match,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              &udp_fwd,
                                              nullptr,
                                              &(root_udp_entry_default));
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Root pipe UDP entry creation failed with: {}",
                            doca_error_get_descr(result));
      return result;
    }

    result = doca_flow_entries_process(df_port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Root pipe entry process failed with: {}",
                            doca_error_get_descr(result));
      return result;
    }
  }

  HOLOSCAN_LOG_INFO("Created Pipe {}", pipe_name);

  return DOCA_SUCCESS;
}

DocaMgr::~DocaMgr() {
  for (auto& rx : cfg_.rx_) {
    for (auto& q : rx.queues_) {
      auto q_backend = static_cast<DocaRxQueue*>(q.common_.backend_config_);
      q_backend->destroy_semaphore();
    }
  }

  /* Tx burst preallocate */
  for (int idx = 0; idx < MAX_TX_BURST; idx++) cudaFree(burst[idx].gpu_pkts_len);
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////

void DocaMgr::run() {
  RxDocaWorkerParams* params_rx;
  TxDocaWorkerParams* params_tx;
  uint32_t lcore_rx = rte_get_next_lcore(-1, 1, 0);
  uint32_t lcore_tx = rte_get_next_lcore(lcore_rx, 1, 0);

  HOLOSCAN_LOG_INFO("Starting advanced network GPU workers");  // rx.empty {}", rx.empty);
  // determine the correct process types for input/output
  int (*rx_worker)(void*) = rx_core;
  int (*tx_worker)(void*) = tx_core;

  for (auto& rx : cfg_.rx_) {
    params_rx = new RxDocaWorkerParams;
    params_rx->port = rx.port_id_;
    params_rx->rxqn = cfg_.rx_[0].queues_.size();
    params_rx->ring = rx_ring;
    params_rx->meta_pool = rx_meta;
    params_rx->gdev = gdev;
    int ridx = 0;
    for (auto& q : rx.queues_) {
      auto qinfo = static_cast<DocaRxQueue*>(q.common_.backend_config_);
      params_rx->rxqw[ridx].queue = q.common_.id_;
      params_rx->rxqw[ridx].batch_size = q.common_.batch_size_;
      params_rx->rxqw[ridx].gpu_id = q.common_.gpu_dev_;
      params_rx->rxqw[ridx].rxq = qinfo;
      HOLOSCAN_LOG_INFO("Queue {} CPU core {}", ridx, lcore_rx);
      ridx++;
    }

    rte_eal_remote_launch(rx_worker, (void*)params_rx, lcore_rx);
  }

  for (auto& tx : cfg_.tx_) {
    params_tx = new TxDocaWorkerParams;
    params_tx->port = tx.port_id_;
    params_tx->txqn = cfg_.tx_[0].queues_.size();
    params_tx->meta_pool = tx_meta;
    params_tx->gdev = gdev;
    rte_eth_macaddr_get(tx.port_id_, &params_tx->mac_addr);
    int tidx = 0;
    for (auto& q : tx.queues_) {
      auto qinfo = static_cast<DocaTxQueue*>(q.common_.backend_config_);
      uint32_t key = (tx.port_id_ << 16) | q.common_.id_;
      params_tx->txqw[tidx].ring = tx_rings[key];
      params_tx->txqw[tidx].queue = q.common_.id_;
      params_tx->txqw[tidx].batch_size = q.common_.batch_size_;
      params_tx->txqw[tidx].gpu_id = q.common_.gpu_dev_;
      params_tx->txqw[tidx].txq = qinfo;
      HOLOSCAN_LOG_INFO("Queue {} CPU core {}", tidx, lcore_tx);
      tidx++;
    }

    rte_eal_remote_launch(tx_worker, (void*)params_tx, lcore_tx);
  }

  HOLOSCAN_LOG_INFO("Done starting workers");
}

////////////////////////////////////////////////////////////////////////////////
///
///  \brief
///
////////////////////////////////////////////////////////////////////////////////

int DocaMgr::rx_core(void* arg) {
  RxDocaWorkerParams* tparams = (RxDocaWorkerParams*)arg;
  int ret = 0;
  uint64_t freq = rte_get_tsc_hz();
  uint64_t timeout_ticks = freq * 0.02;  // expect all packets within 20ms
  uint64_t total_pkts = 0;
  doca_error_t result;
  enum doca_gpu_semaphore_status status;
  cudaStream_t rx_stream;
  cudaError_t res_cuda = cudaSuccess;
  uintptr_t *eth_rxq_cpu_list, *eth_rxq_gpu_list;
  uintptr_t *sem_cpu_list, *sem_gpu_list;
  uint32_t *batch_cpu_list, *batch_gpu_list;
  uint32_t *cpu_exit_condition, *gpu_exit_condition;
  int sem_idx[MAX_NUM_RX_QUEUES] = {0};
  struct adv_doca_rx_gpu_info* packets_stats;
  AdvNetBurstParams* burst;
  CUdevice cuDevice;
  CUcontext cuContext;
  uint64_t last_batch = 0;
  int leastPriority;
  int greatestPriority;

  HOLOSCAN_LOG_INFO("Starting RX Core {}, port {}, queues {}, socket {}",
                    rte_lcore_id(),
                    tparams->port,
                    tparams->rxqn,
                    rte_socket_id());

  /*
   * This is needed in all Holoscan-based applications to run a persistent CUDA kernel.
   * If a persistent CUDA kernel is launched in the default context, no other CUDA kernel
   * is actually executed right after (even on different streams).
   */
  cudaSetDevice(tparams->rxqw[0].gpu_id);  // Need to rely on GPU 0
  cudaFree(0);
  cuDeviceGet(&cuDevice, tparams->rxqw[0].gpu_id);
  cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
  cuCtxPushCurrent(cuContext);

  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uintptr_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&eth_rxq_gpu_list,
                              (void**)&eth_rxq_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory before launching kernel {}",
                       doca_error_get_descr(result));
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uintptr_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&sem_gpu_list,
                              (void**)&sem_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory before launching kernel {}",
                       doca_error_get_descr(result));
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uintptr_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&batch_gpu_list,
                              (void**)&batch_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory before launching kernel {}",
                       doca_error_get_descr(result));
    exit(1);
  }

  for (int idx = 0; idx < tparams->rxqn; idx++) {
    eth_rxq_cpu_list[idx] = (uintptr_t)tparams->rxqw[idx].rxq->eth_rxq_gpu;
    sem_cpu_list[idx] = (uintptr_t)tparams->rxqw[idx].rxq->sem_gpu;
    batch_cpu_list[idx] = tparams->rxqw[idx].batch_size;
  }

  res_cuda = cudaStreamCreateWithPriority(&rx_stream, cudaStreamNonBlocking, greatestPriority);
  if (res_cuda != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Function cudaStreamCreateWithPriority error %d", res_cuda);
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              sizeof(uint32_t),
                              4096,
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void**)&gpu_exit_condition,
                              (void**)&cpu_exit_condition);
  if (result != DOCA_SUCCESS || gpu_exit_condition == nullptr || cpu_exit_condition == nullptr) {
    HOLOSCAN_LOG_ERROR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
    exit(1);
  }
  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

  HOLOSCAN_LOG_INFO("Warmup receive kernel");
  doca_receiver_packet_kernel(
      rx_stream, tparams->rxqn, nullptr, sem_gpu_list, batch_cpu_list, gpu_exit_condition);
  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
  cudaStreamSynchronize(rx_stream);

  HOLOSCAN_LOG_INFO("Launch receive kernel");
  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;
  doca_receiver_packet_kernel(
      rx_stream, tparams->rxqn, eth_rxq_gpu_list, sem_gpu_list, batch_cpu_list, gpu_exit_condition);

  while (!force_quit_doca.load()) {
    for (int ridx = 0; ridx < tparams->rxqn; ridx++) {
      // HOLOSCAN_LOG_INFO("Check queue {} sem {}", ridx, sem_idx[ridx]);
      result =
          doca_gpu_semaphore_get_status(tparams->rxqw[ridx].rxq->sem_cpu, sem_idx[ridx], &status);
      if (result != DOCA_SUCCESS) {
        HOLOSCAN_LOG_ERROR("UDP semaphore error queue {}", ridx);
        force_quit_doca.store(true);
        break;
      }

      if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
        result = doca_gpu_semaphore_get_custom_info_addr(
            tparams->rxqw[ridx].rxq->sem_cpu, sem_idx[ridx], (void**)&(packets_stats));
        if (result != DOCA_SUCCESS) {
          HOLOSCAN_LOG_ERROR("UDP semaphore get address error");
          force_quit_doca.store(true);
          break;
        }

        if (rte_mempool_get(tparams->meta_pool, reinterpret_cast<void**>(&burst)) < 0) {
          HOLOSCAN_LOG_ERROR("Processing function falling behind. No free buffers for metadata!");
          force_quit_doca.store(true);
        }

        //  Queue ID for receiver to differentiate
        burst->hdr.hdr.q_id = tparams->rxqw[ridx].queue;
        burst->hdr.hdr.first_pkt_addr = (uintptr_t)tparams->rxqw[ridx].rxq->gpu_pkt_addr;
        burst->hdr.hdr.max_pkt = tparams->rxqw[ridx].rxq->max_pkt_num;
        burst->hdr.hdr.max_pkt_size = tparams->rxqw[ridx].rxq->max_pkt_size;
        burst->hdr.hdr.port_id = tparams->port;
        burst->hdr.hdr.num_pkts = packets_stats->num_pkts;
        burst->hdr.hdr.nbytes = packets_stats->nbytes;
        burst->hdr.hdr.gpu_pkt0_idx = packets_stats->gpu_pkt0_idx;
        burst->hdr.hdr.gpu_pkt0_addr = packets_stats->gpu_pkt0_addr;
        // HOLOSCAN_LOG_DEBUG("sem {} queue {} num_pkts {}", sem_idx[ridx], ridx,
        // burst->hdr.hdr.num_pkts); Assuming each batch is accumulated by the kernel
        rte_ring_enqueue(tparams->ring, reinterpret_cast<void*>(burst));

        result = doca_gpu_semaphore_set_status(
            tparams->rxqw[ridx].rxq->sem_cpu, sem_idx[ridx], DOCA_GPU_SEMAPHORE_STATUS_FREE);
        if (result != DOCA_SUCCESS) {
          HOLOSCAN_LOG_ERROR("UDP semaphore error queue {}", ridx);
          force_quit_doca.store(true);
          break;
        }

        sem_idx[ridx] = (sem_idx[ridx] + 1) % MAX_DEFAULT_SEM_X_QUEUE;

        total_pkts += burst->hdr.hdr.num_pkts;
        stats_rx_tot_pkts += burst->hdr.hdr.num_pkts;
        stats_rx_tot_bytes += burst->hdr.hdr.nbytes;
        stats_rx_tot_batch++;
      }
    }
  }

  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
  HOLOSCAN_LOG_INFO("Wait receive kernel completion");
  cudaStreamSynchronize(rx_stream);

  for (int ridx = 0; ridx < tparams->rxqn; ridx++) {
    // HOLOSCAN_LOG_INFO("Check queue {} sem {}", ridx, sem_idx[ridx]);
    doca_gpu_semaphore_get_status(tparams->rxqw[ridx].rxq->sem_cpu, sem_idx[ridx], &status);
    if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
      doca_gpu_semaphore_get_custom_info_addr(
          tparams->rxqw[ridx].rxq->sem_cpu, sem_idx[ridx], (void**)&(packets_stats));
      last_batch += packets_stats->num_pkts;
      stats_rx_tot_pkts += packets_stats->num_pkts;
      stats_rx_tot_bytes += packets_stats->nbytes;
      stats_rx_tot_batch++;
    }
  }

  doca_gpu_mem_free(tparams->gdev, (void*)eth_rxq_gpu_list);
  doca_gpu_mem_free(tparams->gdev, (void*)sem_gpu_list);
  cudaStreamDestroy(rx_stream);
  doca_gpu_mem_free(tparams->gdev, (void*)gpu_exit_condition);

  HOLOSCAN_LOG_INFO(
      "Total packets received by application (port/queue {}/{}): {}, last partial batch packets "
      "{}\n",
      tparams->port,
      tparams->rxqn,
      total_pkts + last_batch,
      last_batch);

  cuCtxPopCurrent(&cuContext);

  return 0;
}

int DocaMgr::tx_core(void* arg) {
  TxDocaWorkerParams* tparams = (TxDocaWorkerParams*)arg;
  int ret = 0;
  uint64_t freq = rte_get_tsc_hz();
  uint64_t timeout_ticks = freq * 0.02;  // expect all packets within 20ms
  doca_error_t result;
  enum doca_gpu_semaphore_status status;
  cudaStream_t tx_stream[MAX_DEFAULT_QUEUES];
  cudaError_t res_cuda = cudaSuccess;
  AdvNetBurstParams* burst;
  uint64_t cnt_pkts[MAX_DEFAULT_QUEUES] = {0};
  bool set_completion[MAX_DEFAULT_QUEUES] = {false};

  HOLOSCAN_LOG_INFO("Starting Tx Core {}, port {}, queues {}, socket {}",
                    rte_lcore_id(),
                    tparams->port,
                    tparams->txqn,
                    rte_socket_id());

#if 0
    CUdevice cuDevice;
    CUcontext cuContext;
    cudaSetDevice(tparams->txqw[0].gpu_id);
    cudaFree(0);
    cuDeviceGet(&cuDevice, tparams->txqw[0].gpu_id);
    cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
    cuCtxPushCurrent(cuContext);
#endif
  for (int idxq = 0; idxq < tparams->txqn; idxq++) {
    res_cuda = cudaStreamCreateWithFlags(&tx_stream[idxq], cudaStreamNonBlocking);
    if (res_cuda != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Function cudaStreamCreateWithFlags error %d", res_cuda);
      exit(1);
    }
    HOLOSCAN_LOG_DEBUG("Warmup send kernel queue {}", idxq);
    doca_sender_packet_kernel(
        tx_stream[idxq], tparams->txqw[idxq].txq->eth_txq_gpu, nullptr, 0, 0, 0, 0, false);
    cudaStreamSynchronize(tx_stream[idxq]);
  }

  while (!force_quit_doca.load()) {
    for (int idxq = 0; idxq < tparams->txqn; idxq++) {
      /* Guardrail to prevent issues caused on ARM by the communication between application and
       * operator */
      if (tparams->txqw[idxq].txq->tx_cmp_posted.load() > TX_COMP_THRS) {
        HOLOSCAN_LOG_DEBUG("Queue {} pkts {} too many cmp {}",
                           idxq,
                           cnt_pkts[idxq],
                           tparams->txqw[idxq].txq->tx_cmp_posted.load());
        continue;
      }

      if (rte_ring_dequeue(tparams->txqw[idxq].ring, reinterpret_cast<void**>(&burst)) != 0)
        continue;

      if (idxq != burst->hdr.hdr.q_id)
        HOLOSCAN_LOG_ERROR("Burst queue {} is different from queue id {}. It should not happen!",
                           burst->hdr.hdr.q_id,
                           idxq);

      cudaStreamWaitEvent(tx_stream[idxq], burst->event, 0);

      /* Only necessary checks to prioritize the launch of the kernel */
      cnt_pkts[idxq] += burst->hdr.hdr.num_pkts;
      if (cnt_pkts[idxq] > MAX_SQ_DESCR_NUM / 4) set_completion[idxq] = true;

      doca_sender_packet_kernel(tx_stream[idxq],
                                tparams->txqw[idxq].txq->eth_txq_gpu,
                                tparams->txqw[idxq].txq->buf_arr_gpu,
                                burst->hdr.hdr.gpu_pkt0_idx,
                                burst->hdr.hdr.num_pkts,
                                burst->hdr.hdr.max_pkt,
                                burst->gpu_pkts_len,
                                set_completion[idxq]);

      rte_mempool_put(tparams->meta_pool, burst);

      stats_tx_tot_pkts += burst->hdr.hdr.num_pkts;
      // stats_tx_tot_bytes += burst->hdr.hdr.nbytes;
      stats_tx_tot_batch++;

      /* Remaining checks after kernel launch */
      if (set_completion[idxq] == true) {
        tparams->txqw[idxq].txq->tx_cmp_posted++;
        HOLOSCAN_LOG_DEBUG("Queue {} pkts {} posted cmp {}",
                           idxq,
                           cnt_pkts[idxq],
                           tparams->txqw[idxq].txq->tx_cmp_posted.load());
        cnt_pkts[idxq] = 0;
        set_completion[idxq] = false;
      }
    }
  }

  for (int idxq = 0; idxq < tparams->txqn; idxq++) {
    res_cuda = cudaStreamDestroy(tx_stream[idxq]);
    if (res_cuda != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Function cudaStreamDestroy error %d", res_cuda);
    }
  }

#if 0
    cuCtxPopCurrent(&cuContext);
#endif

  return 0;
}

/* ANO INTERFACE TO BE REMOVED */
/* ANO interface implementations */

void* DocaMgr::get_cpu_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  //   return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]), void*);
  return nullptr;
}

void* DocaMgr::get_gpu_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  uint32_t pkt = burst->hdr.hdr.gpu_pkt0_idx + idx;

  // HOLOSCAN_LOG_INFO("get_gpu_pkt_ptr pkt {} gpu_pkt0_idx {} idx {} addr {}\n",
  //         pkt, burst->hdr.hdr.gpu_pkt0_idx, idx, burst->hdr.hdr.gpu_pkt0_addr);

  if (pkt < burst->hdr.hdr.max_pkt)
    return (void*)(((uintptr_t)burst->hdr.hdr.gpu_pkt0_addr) + (idx * burst->hdr.hdr.max_pkt_size));
  else
    return (void*)(((uintptr_t)burst->hdr.hdr.first_pkt_addr) +
                   ((pkt % burst->hdr.hdr.max_pkt) * burst->hdr.hdr.max_pkt_size));
}

uint64_t DocaMgr::get_burst_tot_byte(AdvNetBurstParams* burst) {
  return burst->hdr.hdr.nbytes;
}

uint16_t DocaMgr::get_cpu_pkt_len(AdvNetBurstParams* burst, int idx) {
  //   return reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx])->data_len;
  return 0;
}

uint16_t DocaMgr::get_gpu_pkt_len(AdvNetBurstParams* burst, int idx) {
  //   return reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx])->data_len;
  return 0;
}

AdvNetStatus DocaMgr::set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp) {
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DocaMgr::get_tx_pkt_burst(AdvNetBurstParams* burst) {
  int buf_idx = 0;

  // Check if burst->hdr.hdr.num_pkts > max_tx_batch_size

  for (auto& tx : cfg_.tx_) {
    for (auto& q : tx.queues_) {
      if (q.common_.id_ == burst->hdr.hdr.q_id) {
        DocaTxQueue* txq = (DocaTxQueue*)q.common_.backend_config_;
        // Should be thread safe as it's atomic inc
        auto buf_idx = txq->buff_arr_idx.fetch_add(burst->hdr.hdr.num_pkts);
        burst->hdr.hdr.max_pkt = txq->max_pkt_num;
        buf_idx = (buf_idx % txq->max_pkt_num);
        burst->hdr.hdr.gpu_pkt0_addr =
            (uintptr_t)((uint8_t*)txq->gpu_pkt_addr + (buf_idx * txq->max_pkt_size));
        burst->hdr.hdr.first_pkt_addr = (uintptr_t)txq->gpu_pkt_addr;
        burst->hdr.hdr.gpu_pkt0_idx = buf_idx;

        HOLOSCAN_LOG_DEBUG(
            "Get TX burst for queue {} ({}) on port {} pkts {} first {} gpu_pkt0_idx {}",
            q.common_.name_,
            q.common_.id_,
            tx.port_id_,
            burst->hdr.hdr.num_pkts,
            burst->hdr.hdr.first_pkt_addr,
            burst->hdr.hdr.gpu_pkt0_idx);
      }
    }
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DocaMgr::set_cpu_eth_hdr(AdvNetBurstParams* burst, int idx, uint8_t* dst_addr) {
  return AdvNetStatus::NOT_SUPPORTED;
}

AdvNetStatus DocaMgr::set_cpu_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                       unsigned int src_host, unsigned int dst_host) {
  return AdvNetStatus::NOT_SUPPORTED;
}

AdvNetStatus DocaMgr::set_cpu_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len,
                                      uint16_t src_port, uint16_t dst_port) {
  return AdvNetStatus::NOT_SUPPORTED;
}

AdvNetStatus DocaMgr::set_cpu_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) {
  return AdvNetStatus::NOT_SUPPORTED;
}

bool DocaMgr::tx_burst_available(AdvNetBurstParams* burst) {
  for (auto& tx : cfg_.tx_) {
    for (auto& q : tx.queues_) {
      if (q.common_.id_ == burst->hdr.hdr.q_id) {
        DocaTxQueue* txq = (DocaTxQueue*)q.common_.backend_config_;
        doca_pe_progress(txq->pe);
        if (txq->tx_cmp_posted > TX_COMP_THRS) {
          HOLOSCAN_LOG_DEBUG("txq->tx_cmp_posted {}", txq->tx_cmp_posted);
          return false;
        }
      }
    }
  }

  return true;
}

AdvNetStatus DocaMgr::set_pkt_len(AdvNetBurstParams* burst, int idx, int cpu_len, int gpu_len) {
  burst->gpu_pkts_len[idx] = gpu_len;
  return AdvNetStatus::SUCCESS;
}

void DocaMgr::free_pkt(void* pkt) {
  return;
}

void DocaMgr::free_pkts(void** pkts, int num_pkts) {
  return;
}

void DocaMgr::free_rx_burst(AdvNetBurstParams* burst) {
  return;
}

void DocaMgr::free_tx_burst(AdvNetBurstParams* burst) {
  return;
}

std::optional<uint16_t> DocaMgr::get_port_from_ifname(const std::string& name) {
  uint16_t port;
  doca_error_t result;

  HOLOSCAN_LOG_INFO("Port name {}", name);
  result = doca_dpdk_get_first_port_id(ddev_rx, &port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Function doca_dpdk_get_first_port_id returned %s",
                          doca_error_get_descr(result));
    return {};
  }
  // auto ret = rte_eth_dev_get_port_by_name(name.c_str(), &port);

  return port;
}

AdvNetStatus DocaMgr::get_rx_burst(AdvNetBurstParams** burst) {
  if (rte_ring_dequeue(rx_ring, reinterpret_cast<void**>(burst)) < 0) {
    return AdvNetStatus::NOT_READY;
  }

  return AdvNetStatus::SUCCESS;
}

void DocaMgr::free_rx_meta(AdvNetBurstParams* burst) {
  rte_mempool_put(rx_meta, burst);
}

void DocaMgr::free_tx_meta(AdvNetBurstParams* burst) {
  rte_mempool_put(tx_meta, burst);
}

AdvNetBurstParams* DocaMgr::create_burst_params() {
  auto burst_idx = burst_tx_idx.fetch_add(1);
  HOLOSCAN_LOG_DEBUG(
      "create_burst_params burst_idx {} MAX_TX_BURST {}", burst_idx % MAX_TX_BURST, MAX_TX_BURST);
  return &(burst[burst_idx % MAX_TX_BURST]);
}

AdvNetStatus DocaMgr::get_tx_meta_buf(AdvNetBurstParams** burst) {
  if (rte_mempool_get(tx_meta, reinterpret_cast<void**>(burst)) != 0) {
    fprintf(stderr, "Failed to get TX meta descriptor\n");
    HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor");
    return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus DocaMgr::send_tx_burst(AdvNetBurstParams* burst) {
  uint32_t key = (burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id;
  const auto ring = tx_rings.find(key);

  if (ring == tx_rings.end()) {
    HOLOSCAN_LOG_ERROR("Invalid port/queue combination in send_tx_burst: {}/{}",
                       burst->hdr.hdr.port_id,
                       burst->hdr.hdr.q_id);
    return AdvNetStatus::INVALID_PARAMETER;
  }

  if (rte_ring_enqueue(ring->second, reinterpret_cast<void*>(burst)) != 0) {
    fprintf(stderr, "calling DOCA free_tx_meta\n");
    free_tx_meta(burst);
    HOLOSCAN_LOG_CRITICAL("Failed to enqueue TX work");
    return AdvNetStatus::NO_SPACE_AVAILABLE;
  }

  return AdvNetStatus::SUCCESS;
}

void DocaMgr::shutdown() {
  int icore = 0;

  HOLOSCAN_LOG_INFO("ANO DOCA manager shutting down");

  force_quit_doca.store(true);

  HOLOSCAN_LOG_INFO("ANO DOCA manager stopping cores");

  RTE_LCORE_FOREACH_WORKER(icore) {
    if (rte_eal_wait_lcore(icore) < 0) {
      fprintf(stderr, "bad exit for coreid: %d\n", icore);
      break;
    }
  }
}

void DocaMgr::print_stats() {
  HOLOSCAN_LOG_INFO("ANO DOCA manager stats");
  HOLOSCAN_LOG_INFO("Total Rx packets {}", stats_rx_tot_pkts);
  HOLOSCAN_LOG_INFO("Total Rx bytes {}", stats_rx_tot_bytes);
  HOLOSCAN_LOG_INFO("Total Rx batch processed {}", stats_rx_tot_batch);

  HOLOSCAN_LOG_INFO("Total Tx packets {}", stats_tx_tot_pkts);
  HOLOSCAN_LOG_INFO("Total Tx bytes {}", stats_tx_tot_bytes);
  HOLOSCAN_LOG_INFO("Total Tx batch processed {}", stats_tx_tot_batch);

  return;
}

};  // namespace holoscan::ops
