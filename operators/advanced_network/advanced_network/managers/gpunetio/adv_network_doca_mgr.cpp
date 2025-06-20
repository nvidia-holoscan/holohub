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
#include <thread>
#include <unordered_map>
#include "adv_network_doca_mgr.h"
#include "adv_network_doca_kernels.h"
#include "holoscan/holoscan.hpp"

using namespace std::chrono;

namespace holoscan::advanced_network {

// --- Local Helper Functions for Port/Queue Key Management ---

/**
 * @brief Generates a unique 32-bit key from a port and queue ID.
 * Assumes port ID fits in the upper 16 bits and queue ID in the lower 16 bits.
 *
 * @param port_id The port ID.
 * @param queue_id The queue ID.
 * @return uint32_t The combined key.
 */
static inline uint32_t generate_queue_key(int port_id, int queue_id) {
  return (static_cast<uint32_t>(port_id) << 16) | static_cast<uint32_t>(queue_id);
}

/**
 * @brief Extracts the port ID from a 32-bit queue key.
 *
 * @param key The combined key.
 * @return int The extracted port ID.
 */
static inline int get_port_from_key(uint32_t key) {
  return static_cast<int>((key >> 16) & 0xFFFF);
}

/**
 * @brief Extracts the queue ID from a 32-bit queue key.
 *
 * @param key The combined key.
 * @return int The extracted queue ID.
 */
static inline int get_queue_from_key(uint32_t key) {
  return static_cast<int>(key & 0xFFFF);
}

// --- End Helper Functions ---

DocaMgr doca_mgr{};

std::atomic<bool> force_quit_doca = false;
uint64_t stats_rx_tot_pkts;
uint64_t stats_rx_tot_bytes;
uint64_t stats_rx_tot_batch;

uint64_t stats_tx_tot_pkts;
uint64_t stats_tx_tot_bytes;
uint64_t stats_tx_tot_batch;

struct TxDocaWorkerQueue {
  int port;
  int queue;
  uint64_t tx_pkts = 0;
  uint32_t batch_size;
  struct rte_ring* ring;
  DocaTxQueue* txq;
};

struct TxDocaWorkerParams {
  int core_id;
  int txqn;
  int gpu_id;
  struct doca_gpu* gdev;
  struct rte_mempool* meta_pool;
  struct rte_ether_addr mac_addr;
  struct TxDocaWorkerQueue txqw[MAX_NUM_TX_QUEUES];
};

struct RxDocaWorkerQueue {
  int port;
  int queue;
  uint64_t rx_pkts = 0;
  uint32_t batch_size;
  DocaRxQueue* rxq;
  struct rte_ring* ring;
};

struct RxDocaWorkerParams {
  int core_id;
  int rxqn;
  int gpu_id;
  struct doca_gpu* gdev;
  struct rte_mempool* meta_pool;
  struct RxDocaWorkerQueue rxqw[MAX_NUM_RX_QUEUES];
};

const std::unordered_map<LogLevel::Level, doca_log_level>
    DocaLogLevel::adv_net_to_doca_log_level_map = {
        {LogLevel::TRACE, DOCA_LOG_LEVEL_TRACE},
        {LogLevel::DEBUG, DOCA_LOG_LEVEL_DEBUG},
        {LogLevel::INFO, DOCA_LOG_LEVEL_INFO},
        {LogLevel::WARN, DOCA_LOG_LEVEL_WARNING},
        {LogLevel::ERROR, DOCA_LOG_LEVEL_ERROR},
        {LogLevel::CRITICAL, DOCA_LOG_LEVEL_CRIT},
        {LogLevel::OFF, DOCA_LOG_LEVEL_DISABLE},
};

const std::unordered_map<doca_log_level, std::string>
    DocaLogLevel::level_to_string_description_map = {
        {DOCA_LOG_LEVEL_TRACE, "Trace"},
        {DOCA_LOG_LEVEL_DEBUG, "Debug"},
        {DOCA_LOG_LEVEL_INFO, "Info"},
        {DOCA_LOG_LEVEL_WARNING, "Warning"},
        {DOCA_LOG_LEVEL_ERROR, "Error"},
        {DOCA_LOG_LEVEL_CRIT, "Critical"},
        {DOCA_LOG_LEVEL_DISABLE, "Disable"},
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
    HOLOSCAN_LOG_ERROR("Failed to load doca devices list. Doca_error value: {}",
                       static_cast<int>(res));
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

doca_error_t DocaMgr::init_doca_devices() {
  doca_error_t result;
  constexpr int max_nargs = 32;
  constexpr int max_arg_size = 64;
  int ret;
  char** argv_;
  int arg = 0;
  argv_ = (char**)malloc(sizeof(char*) * max_nargs);
  for (int i = 0; i < max_nargs; i++) argv_[i] = (char*)calloc(max_arg_size, sizeof(char));

  std::string cores = std::to_string(cfg_.common_.master_core_) + ",";  // Master core must be first

  for (const auto& intf : cfg_.ifs_) {
    for (const auto& q : intf.rx_.queues_) { cores += q.common_.cpu_core_ + ","; }

    for (const auto& q : intf.tx_.queues_) { cores += q.common_.cpu_core_ + ","; }
  }

  cores = cores.substr(0, cores.size() - 1);
  std::cout << cores;

  // strncpy(argv_[arg++], "", max_arg_size - 1);
  // strncpy(argv_[arg++], "-l", max_arg_size - 1);
  // strncpy(argv_[arg++], cores.c_str(), max_arg_size - 1);
  strncpy(argv_[arg++], "", max_arg_size - 1);
  strncpy(argv_[arg++], "-a", max_arg_size - 1);
  strncpy(argv_[arg++], std::string("00:00.0").c_str(), max_arg_size - 1);

  HOLOSCAN_LOG_INFO("Initializing DPDK on cores {} max_nargs {} arg {} : {} {} {} {}",
                    cores.c_str(),
                    max_nargs,
                    arg,
                    argv_[0],
                    argv_[1],
                    argv_[2],
                    argv_[3]);
  ret = rte_eal_init(arg, argv_);
  if (ret < 0) {
    HOLOSCAN_LOG_CRITICAL("DPDK init failed: {}", ret);
    return DOCA_ERROR_DRIVER;
  }

  int port_id = 0;
  for (auto& intf : cfg_.ifs_) {
    // Assign an arbitrary port ID in the interface config for faster lookup
    intf.port_id_ = port_id++;
    HOLOSCAN_LOG_INFO("Initializing interface {} ({} - port {})",
                      intf.name_, intf.address_, intf.port_id_);
    result = open_doca_device_with_pci(intf.address_.c_str(), &ddev[intf.port_id_]);
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to open NIC device based on PCI address");
      return result;
    }

    /* Enable DOCA Flow HWS mode */
    result = doca_dpdk_port_probe(ddev[intf.port_id_], "dv_flow_en=2");
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Function doca_dpdk_port_probe returned {}",
                            doca_error_get_descr(result));
      return result;
    }

    rte_eth_macaddr_get(intf.port_id_, &mac_addrs[intf.port_id_]);
    HOLOSCAN_LOG_INFO("DOCA init Port {} -- RX: {} TX: {}",
                      intf.port_id_,
                      intf.rx_.queues_.size() > 0 ? "ENABLED" : "DISABLED",
                      intf.tx_.queues_.size() > 0 ? "ENABLED" : "DISABLED");
  }

  auto log_level = DocaLogLevel::from_adv_net_log_level(cfg_.log_level_);
  if (log_level != DOCA_LOG_LEVEL_DISABLE) {
    struct doca_log_backend* stdout_logger = nullptr;

    HOLOSCAN_LOG_INFO("Setting DOCA Logging level to {}",
                      DocaLogLevel::to_description_string(log_level));

    result = doca_log_backend_create_with_file_sdk(stdout, &stdout_logger);
    if (result != DOCA_SUCCESS) return result;

    result = doca_log_backend_set_sdk_level(stdout_logger, log_level);
    if (result != DOCA_SUCCESS) return result;
  }

  return DOCA_SUCCESS;
}

struct doca_flow_port* DocaMgr::init_doca_flow(uint16_t port_id, uint8_t rxq_num) {
  doca_error_t result;
  char port_id_str[MAX_PORT_STR_LEN];
  struct doca_flow_port_cfg* port_cfg;
  struct doca_flow_port* df_port;
  struct doca_flow_cfg* rxq_flow_cfg;
  static bool flow_init = false;
  doca_error_t ret = DOCA_SUCCESS;
  int ret_dpdk = 0;
  struct rte_eth_dev_info dev_info = {0};
  struct rte_eth_conf eth_conf = {
      .rxmode = {
              .mtu = 2048, /* Not really used, just to initialize DPDK */
          },
  };
  struct rte_flow_error error;

  HOLOSCAN_LOG_INFO("Initializing DOCA flow on port {} with {} queues", port_id, rxq_num);
  /*
   * DPDK should be initialized and started before DOCA Flow.
   * DPDK doesn't start the device without, at least, one DPDK Rx queue.
   * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
   *
   * Following lines of code can be considered the minimum WAR for this issue.
   */

  ret_dpdk = rte_eth_dev_info_get(port_id, &dev_info);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_info_get with: {}", rte_strerror(-ret));
    return nullptr;
  }

  ret_dpdk = rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_configure with: {}", rte_strerror(-ret));
    return nullptr;
  }

  for (int idx = 0; idx < rxq_num; idx++) {
    struct rte_mempool* mp = nullptr;
    std::string name =
        std::string("RX_POOL_P") + std::to_string(port_id) + "_Q" + std::to_string(idx);
    mp = rte_pktmbuf_pool_create(name.c_str(), 8192, 0, 0, 8192, rte_eth_dev_socket_id(port_id));
    if (mp == nullptr) {
      HOLOSCAN_LOG_CRITICAL("Failed rte_pktmbuf_pool_create with: {}", rte_strerror(-ret));
      return nullptr;
    }

    ret_dpdk =
        rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), nullptr, mp);
    if (ret) {
      HOLOSCAN_LOG_CRITICAL("Failed rte_eth_rx_queue_setup with: {}", rte_strerror(-ret));
      return nullptr;
    }
  }

  ret_dpdk = rte_flow_isolate(port_id, 1, &error);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_flow_isolate with: {}", error.message);
    return nullptr;
  }

  ret_dpdk = rte_eth_dev_start(port_id);
  if (ret) {
    HOLOSCAN_LOG_CRITICAL("Failed rte_eth_dev_start with: {}", rte_strerror(-ret));
    return nullptr;
  }

  if (!flow_init) {
    /* Initialize doca flow framework */
    ret = doca_flow_cfg_create(&rxq_flow_cfg);
    if (ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to create doca_flow_cfg: {}", doca_error_get_descr(ret));
      return nullptr;
    }

    ret = doca_flow_cfg_set_pipe_queues(rxq_flow_cfg, rxq_num);
    if (ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg pipe_queues: {}",
                            doca_error_get_descr(ret));
      doca_flow_cfg_destroy(rxq_flow_cfg);
      return nullptr;
    }

    /*
     * HWS: Hardware steering
     * Isolated: don't create RSS rule for DPDK created RX queues
     */
    ret = doca_flow_cfg_set_mode_args(rxq_flow_cfg, "vnf,hws,isolated");
    if (ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg mode_args: {}", doca_error_get_descr(ret));
      doca_flow_cfg_destroy(rxq_flow_cfg);
      return nullptr;
    }

    ret = doca_flow_cfg_set_nr_counters(rxq_flow_cfg, FLOW_NB_COUNTERS);
    if (ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_cfg nr_counters: {}",
                            doca_error_get_descr(ret));
      doca_flow_cfg_destroy(rxq_flow_cfg);
      return nullptr;
    }

    result = doca_flow_init(rxq_flow_cfg);
    if (result != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL(
          "Failed to init doca flow with: {}:{}", (int)result, doca_error_get_descr(result));
      return nullptr;
    }

    doca_flow_cfg_destroy(rxq_flow_cfg);

    flow_init = true;
  }

  /* Start doca flow port */
  result = doca_flow_port_cfg_create(&port_cfg);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to create doca_flow_port_cfg: {}", doca_error_get_descr(result));
    return nullptr;
  }
  snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_id);
  result = doca_flow_port_cfg_set_devargs(port_cfg, port_id_str);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to set doca_flow_port_cfg devargs: {}",
                          doca_error_get_descr(result));
    doca_flow_port_cfg_destroy(port_cfg);
    return nullptr;
  }
  result = doca_flow_port_start(port_cfg, &df_port);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed to start doca flow port with: {}", doca_error_get_descr(result));
    doca_flow_port_cfg_destroy(port_cfg);
    return nullptr;
  }
  doca_flow_port_cfg_destroy(port_cfg);

  HOLOSCAN_LOG_INFO("Successfully started DOCA flow for port {}", port_id);

  return df_port;
}

int DocaMgr::setup_pools_and_rings(int max_tx_batch) {
  BurstParams* bursts_rx[(1U << 6) - 1U];
  BurstParams* bursts_tx[(1U << 7) - 1U];
  int idx = 0;

  HOLOSCAN_LOG_DEBUG("Setting up RX meta pool");
  rx_metadata = rte_mempool_create("RX_META_POOL",
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
  if (rx_metadata == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate RX meta pool!");
    return -1;
  }

  while (idx < (1U << 6) - 1U &&
         rte_mempool_get(rx_metadata, reinterpret_cast<void**>(&bursts_rx[idx])) == 0) {
    bursts_rx[idx]->pkts[0] = (void**)calloc(CUDA_MAX_RX_NUM_PKTS, sizeof(void*));
    idx++;
  }

  rte_mempool_put_bulk(rx_metadata, reinterpret_cast<void**>(&bursts_rx), idx);

  // Create rings per queue
  for (const auto& intf : cfg_.ifs_) {
    for (const auto& q : intf.rx_.queues_) {
      uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
      auto name =
          "RX_RING_P" + std::to_string(intf.port_id_) + "_Q" + std::to_string(q.common_.id_);
      HOLOSCAN_LOG_INFO("Setting up RX ring {}", name);
      rx_rings[key] = rte_ring_create(
          name.c_str(), 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
      if (rx_rings[key] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate ring {}!", name);
        return -1;
      }
    }

    for (const auto& q : intf.tx_.queues_) {
      const auto append =
          "P" + std::to_string(intf.port_id_) + "_Q" + std::to_string(q.common_.id_);
      auto name = "TX_RING_" + append;
      HOLOSCAN_LOG_INFO("Setting up TX ring {}", name);
      uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
      tx_rings[key] = rte_ring_create(
          name.c_str(), 2048, rte_socket_id(), RING_F_MC_RTS_DEQ | RING_F_MP_RTS_ENQ);
      if (tx_rings[key] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Failed to allocate ring!");
        return -1;
      }
    }
  }

  HOLOSCAN_LOG_INFO("Setting up TX meta pool");
  tx_metadata = rte_mempool_create("TX_META_POOL",
                                   (1U << 7) - 1U,
                                   sizeof(BurstParams),
                                   0,
                                   0,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   rte_socket_id(),
                                   0);
  if (tx_metadata == nullptr) {
    HOLOSCAN_LOG_CRITICAL("Failed to allocate TX meta pool!");
    return -1;
  }

  idx = 0;
  while (idx < (1U << 7) - 1U &&
         rte_mempool_get(tx_metadata, reinterpret_cast<void**>(&bursts_tx[idx])) == 0) {
    bursts_tx[idx]->pkts[0] = (void**)calloc(1, sizeof(void*));
    cudaMallocHost(&bursts_tx[idx]->pkt_lens[0], max_tx_batch * sizeof(uint32_t));
    memset(bursts_tx[idx]->pkt_lens[0], 0, max_tx_batch * sizeof(uint32_t));
    idx++;
  }

  rte_mempool_put_bulk(tx_metadata, reinterpret_cast<void**>(&bursts_tx), idx);

  return 0;
}

bool DocaMgr::set_config_and_initialize(const NetworkConfig& cfg) {
  if (!this->initialized_) {
    cfg_ = cfg;
    cpu_set_t mask;
    long nproc, i;

    if (!validate_config()) {
      HOLOSCAN_LOG_CRITICAL("Config validation failed");
      return false;
    }

    // Start Initialize in a separate thread so it doesn't set the affinity for the
    // whole application
    std::thread t(&DocaMgr::initialize, this);
    t.join();

    this->initialized_ = true;

    run();
  }

  return true;
}

bool DocaMgr::validate_config() const {
  if (!Manager::validate_config()) { return false; }

  // Don't allow buffer splitting
  for (const auto& intf : cfg_.ifs_) {
    int gpu_id = -1;

    for (const auto& rxq : intf.rx_.queues_) {
      if (gpu_id == -1) {
        gpu_id = cfg_.mrs_.at(rxq.common_.mrs_[0]).affinity_;
      } else {
        if (gpu_id != cfg_.mrs_.at(rxq.common_.mrs_[0]).affinity_) {
          HOLOSCAN_LOG_ERROR("GPU comms requires all queue MRs to point to same GPU device");
          return false;
        }
      }

      if (rxq.common_.mrs_.size() > 1) {
        HOLOSCAN_LOG_ERROR("RX buffer split not supported in GPU comms mode yet");
        return false;
      }
    }

    gpu_id = -1;
    for (const auto& txq : intf.rx_.queues_) {
      if (gpu_id == -1) {
        gpu_id = cfg_.mrs_.at(txq.common_.mrs_[0]).affinity_;
      } else {
        if (gpu_id != cfg_.mrs_.at(txq.common_.mrs_[0]).affinity_) {
          HOLOSCAN_LOG_ERROR("GPU comms requires all queue MRs to point to same GPU device");
          return false;
        }
      }

      if (txq.common_.mrs_.size() > 1) {
        HOLOSCAN_LOG_ERROR("Tx buffer split not supported in GPU comms mode yet");
        return false;
      }
    }
  }

  HOLOSCAN_LOG_INFO("Config validated successfully");
  return true;
}

void DocaMgr::initialize() {
  int ret;
  doca_error_t doca_ret;
  /* Initialize DPDK params */
  int max_tx_batch_size = 0;
  size_t max_packet_size = 0;
  enum doca_gpu_mem_type mtype;
  uint32_t key;
  int gpu_id = -1;

  doca_ret = init_doca_devices();
  if (doca_ret != DOCA_SUCCESS) {
    HOLOSCAN_LOG_CRITICAL("Failed init DOCA device {}", net_bdf);
    return;
  }

  // Find all GPUs used in the MRs
  for (const auto& mr : cfg_.mrs_) {
    if (mr.second.kind_ == MemoryKind::DEVICE) { gpu_mr_devs.emplace(mr.second.affinity_); }
  }

  // Populate all GPU device structures
  for (const auto gpu_dev : gpu_mr_devs) {
    char gpu_bdf[MAX_PCIE_STR_LEN];
    if (cudaDeviceGetPCIBusId(gpu_bdf, sizeof(gpu_bdf), gpu_dev) != cudaSuccess) {
      HOLOSCAN_LOG_CRITICAL("Failed get GPU PCIe addr device {}", gpu_dev);
      return;
    }

    doca_ret = doca_gpu_create(gpu_bdf, &gdev[gpu_dev]);
    if (doca_ret != DOCA_SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed get DOCA GPU device {}", gpu_mr_devs);
      return;
    }
  }

  // For now make a single queue. Support more sophisticated TX on next release
  for (auto& intf : cfg_.ifs_) {
    const auto& tx = intf.tx_;
    for (auto& q : tx.queues_) {
      max_tx_batch_size = std::max(max_tx_batch_size, q.common_.batch_size_);

      const auto mr_name = q.common_.mrs_[0];  // Only use the first MR (no splitting)
      for (const auto& mr : cfg_.mrs_) {
        if (mr.first == mr_name) {
          max_packet_size = std::max(max_packet_size, mr.second.buf_size_);
        }
      }
    }
  }

  if (setup_pools_and_rings(max_tx_batch_size) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to set up pools and rings!");
    return;
  }

  for (const auto& intf : cfg_.ifs_) {
    if (intf.rx_.queues_.size() > 0) {
      df_port[intf.port_id_] = init_doca_flow(intf.port_id_, intf.rx_.queues_.size());
      if (df_port[intf.port_id_] == nullptr) {
        HOLOSCAN_LOG_CRITICAL("FAILED: init_doca_flow for port {}", intf.port_id_);
        return;
      }
    }
  }

  // Create DOCA queues
  for (auto& intf : cfg_.ifs_) {
    for (auto& q : intf.rx_.queues_) {
      HOLOSCAN_LOG_INFO("Configuring RX queue: {} ({}) on port {}",
                        q.common_.name_,
                        q.common_.id_,
                        intf.port_id_);
      const auto mr_name = q.common_.mrs_[0];  // Only use the first MR (no splitting)
      size_t q_max_packet_size = 0;
      int rxq_pkts = -1;

      for (const auto& mr : cfg_.mrs_) {
        if (mr.first == mr_name) {
          rxq_pkts = mr.second.num_bufs_;
          q_max_packet_size = mr.second.buf_size_;
          gpu_id = mr.second.affinity_;

          if (mr.second.kind_ == MemoryKind::DEVICE) {
            mtype = DOCA_GPU_MEM_TYPE_GPU;
          } else if (mr.second.kind_ == MemoryKind::HOST_PINNED) {
            mtype = DOCA_GPU_MEM_TYPE_CPU_GPU;
          } else {
            HOLOSCAN_LOG_CRITICAL(
                "FAILED: DOCA mgr doesn't support memory kind different from DEVICE or "
                "HOST_PINNED");
            return;
          }

          break;
        }
      }

      if (!rte_is_power_of_2(rxq_pkts)) { rxq_pkts = rte_align32pow2(rxq_pkts); }

      if (q_max_packet_size > THRESHOLD_PKT_SIZE && rxq_pkts > THRESHOLD_BUF_NUM) {
        HOLOSCAN_LOG_WARN("Decreasing num_bufs to {}", THRESHOLD_BUF_NUM);
        rxq_pkts = THRESHOLD_BUF_NUM;
      }

      if (!rte_is_power_of_2(q_max_packet_size)) {
        q_max_packet_size = rte_align32pow2(q_max_packet_size);
      }

      key = generate_queue_key(intf.port_id_, q.common_.id_);

      HOLOSCAN_LOG_INFO(
          "Configuring RX queue: {} ({}) on port {} memory type {} rxq_pkts {} q_max_packet_size "
          "{}",
          q.common_.name_,
          q.common_.id_,
          intf.port_id_,
          static_cast<int>(mtype),
          rxq_pkts,
          q_max_packet_size);

      rx_q_map_[key] = new DocaRxQueue(ddev[intf.port_id_],
                                       gdev[gpu_id],
                                       df_port[intf.port_id_],
                                       q.common_.id_,
                                       rxq_pkts,
                                       q_max_packet_size,
                                       mtype);
    }

    for (auto& q : intf.tx_.queues_) {
      int txq_pkts = -1;
      key = generate_queue_key(intf.port_id_, q.common_.id_);

      for (const auto& mr : cfg_.mrs_) {
        if (mr.first == q.common_.mrs_[0]) {
          gpu_id = mr.second.affinity_;

          txq_pkts = next_power_of_two(mr.second.num_bufs_);
          if (mr.second.kind_ == MemoryKind::DEVICE) {
            mtype = DOCA_GPU_MEM_TYPE_GPU;
          } else if (mr.second.kind_ == MemoryKind::HOST_PINNED) {
            mtype = DOCA_GPU_MEM_TYPE_CPU_GPU;
          } else {
            HOLOSCAN_LOG_CRITICAL(
                "FAILED: DOCA mgr doesn't support memory kind different from DEVICE or "
                "HOST_PINNED");
            return;
          }
          break;
        }
      }

      HOLOSCAN_LOG_INFO("Configuring TX queue: {} ({}) on port {} memory type {}",
                        q.common_.name_,
                        q.common_.id_,
                        intf.port_id_,
                        static_cast<int>(mtype));

      tx_q_map_[key] = new DocaTxQueue(ddev[intf.port_id_],
                                       gdev[gpu_id],
                                       q.common_.id_,
                                       txq_pkts,
                                       max_packet_size,
                                       mtype,
                                       &decrease_txq_completion_cb);
    }

    if (intf.rx_.queues_.size() > 0) {
      create_default_pipe(intf.port_id_, intf.rx_.queues_.size() - intf.rx_.flows_.size());

      int flow_num = 0;
      for (auto& flow : intf.rx_.flows_) {
        HOLOSCAN_LOG_INFO("Create RX flow {} to queue {}", flow.name_, flow.action_.id_);
        for (auto& q : intf.rx_.queues_) {
          key = generate_queue_key(intf.port_id_, q.common_.id_);
          auto q_backend = rx_q_map_[key];
          if (q_backend->qid == flow.action_.id_) {
            q_backend->create_udp_pipe(flow, rxq_pipe_default);
            flow.backend_config_ = q_backend;
          }
        }
      }

      if (intf.rx_.queues_.size() > 0) {
        doca_ret = create_root_pipe(intf.port_id_);
        if (doca_ret != DOCA_SUCCESS) { HOLOSCAN_LOG_CRITICAL("Can't create UDP root pipe"); }
      }

      /* Create semaphore for GPU - CPU communication per rxq*/
      for (auto& q : cfg_.ifs_[intf.port_id_].rx_.queues_) {
        HOLOSCAN_LOG_INFO("Create RX semaphore");
        key = generate_queue_key(intf.port_id_, q.common_.id_);
        auto q_backend = rx_q_map_[key];
        q_backend->create_semaphore();
      }
    }
  }

  /* Tx burst preallocate */
  for (int idx = 0; idx < MAX_TX_BURST; idx++) {
    cudaMallocHost(&(burst[idx].pkt_lens[0]), max_tx_batch_size * sizeof(uint32_t));
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

doca_error_t DocaMgr::create_default_pipe(int port_id, uint32_t cnt_defq) {
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

  std::string pipe_name = std::string("GPU_RXQ_UDP_DEF_PIPE_P") + std::to_string(port_id);

  if (cnt_defq >= MAX_DEFAULT_QUEUES) {
    HOLOSCAN_LOG_CRITICAL("Too many default queues {}", cnt_defq);
    return DOCA_ERROR_INVALID_VALUE;
  }

  if (cnt_defq == 0) {
    HOLOSCAN_LOG_WARN("No need for a default queue");
    return DOCA_SUCCESS;
  }

  match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
  // match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

  result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port[port_id]);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create doca_flow_pipe_cfg: {}", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name.c_str());
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg name: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg enable_strict_matching: {}",
                       doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg type: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, false);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg is_root: {}",
                       doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, &match_mask);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg match: {}", doca_error_get_descr(result));
    return result;
  }
  result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg monitor: {}",
                       doca_error_get_descr(result));
    return result;
  }

  bool create_pipe;
  const auto& rx = cfg_.ifs_[port_id].rx_;
  for (auto& q : rx.queues_) {
    create_pipe = true;
    uint32_t key = generate_queue_key(port_id, q.common_.id_);
    auto q_backend = rx_q_map_[key];

    for (auto& flow : rx.flows_) {
      if (q_backend->qid == flow.action_.id_) create_pipe = false;
    }

    if (create_pipe == true) {
      // Add default entries
      doca_eth_rxq_get_flow_queue_id(q_backend->eth_rxq_cpu, &flow_queue_id);
      rss_queues[idxq] = flow_queue_id;
      HOLOSCAN_LOG_DEBUG("create_default_pipe idx {} queue {}", idxq, flow_queue_id);
      idxq++;
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

  result = doca_flow_entries_process(df_port[port_id], 0, default_flow_timeout_usec, 0);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("RxQ pipe entry process failed with: {}", doca_error_get_descr(result));
    return result;
  }

  HOLOSCAN_LOG_INFO("Created Default Pipe {}", pipe_name);

  return DOCA_SUCCESS;
}

doca_error_t DocaMgr::create_root_pipe(int port_id) {
  doca_error_t result;
  uint32_t cnt_defq = cfg_.ifs_[port_id].rx_.queues_.size() - cfg_.ifs_[port_id].rx_.flows_.size();

  struct doca_flow_match match_mask = {0};
  struct doca_flow_match udp_match = {0};
  struct doca_flow_monitor monitor = {
      .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
  };
  struct doca_flow_pipe_cfg* pipe_cfg;
  std::string pipe_name = std::string("ROOT_PIPE_P") + std::to_string(port_id);
  memset(&match_mask, 0, sizeof(match_mask));

  result = doca_flow_pipe_cfg_create(&pipe_cfg, df_port[port_id]);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create doca_flow_pipe_cfg: {}", doca_error_get_descr(result));
    return result;
  }

  result = doca_flow_pipe_cfg_set_name(pipe_cfg, pipe_name.c_str());
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg name: {}", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg enable_strict_matching: {}",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_CONTROL);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg type: {}", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, true);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg is_root: {}",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_match(pipe_cfg, nullptr, &match_mask);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg match: {}", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set doca_flow_pipe_cfg monitor: {}",
                       doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }

  result = doca_flow_pipe_create(pipe_cfg, nullptr, nullptr, &root_pipe[port_id]);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Root pipe creation failed with: {}", doca_error_get_descr(result));
    doca_flow_pipe_cfg_destroy(pipe_cfg);
    return result;
  }
  doca_flow_pipe_cfg_destroy(pipe_cfg);

  udp_match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
  udp_match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

  const auto& rx = cfg_.ifs_[port_id].rx_;
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
                                              root_pipe[port_id],
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
                                              root_pipe[port_id],
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

    result = doca_flow_entries_process(df_port[port_id], 0, default_flow_timeout_usec, 0);
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
  // const auto& rx = cfg_.ifs_[0].rx_;
  // for (auto& q : rx.queues_) {
  //   uint32_t key = (cfg_.ifs_[0].port_id_ << 16) | q.common_.id_;
  //   auto q_backend = rx_q_map_[key];
  //   q_backend->destroy_semaphore();
  // }

  /* Tx burst preallocate */
  for (int idx = 0; idx < MAX_TX_BURST; idx++) { cudaFreeHost(burst[idx].pkt_lens[0]); }
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
  cpu_set_t cpuset;

  worker_th_idx = 0;

  HOLOSCAN_LOG_INFO("Starting advanced network GPU workers");  // rx.empty {}", rx.empty);
  // determine the correct process types for input/output
  int (*rx_worker)(void*) = rx_core;
  int (*tx_worker)(void*) = tx_core;
  int ridx = 0, tidx = 0;
  bool rx_enabled = false;
  bool tx_enabled = false;

  for (const auto& intf : cfg_.ifs_) {
    if (intf.rx_.queues_.size() > 0) {
      rx_enabled = true;
      break;
    }
  }

  for (const auto& intf : cfg_.ifs_) {
    if (intf.tx_.queues_.size() > 0) {
      tx_enabled = true;
      break;
    }
  }

  if (rx_enabled) {
    for (const auto gpu_idx : gpu_mr_devs) {
      params_rx = new RxDocaWorkerParams;

      params_rx->meta_pool = rx_metadata;
      params_rx->gpu_id = gpu_idx;  // cfg_.mrs_[rx.queues_[0].common_.mrs_[0]].affinity_;
      params_rx->gdev = gdev[params_rx->gpu_id];
      params_rx->rxqn = 0;

      for (const auto& intf : cfg_.ifs_) {
        const auto& rx = intf.rx_;
        for (auto& q : rx.queues_) {
          if (cfg_.mrs_[q.common_.mrs_[0]].affinity_ == gpu_idx) {
            params_rx->rxqn++;

            if (ridx == 0) params_rx->core_id = stoi(q.common_.cpu_core_);

            uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
            // Assign ring pointer to rxqw
            auto ring_it = rx_rings.find(key);
            if (ring_it == rx_rings.end()) {
              HOLOSCAN_LOG_CRITICAL("Run: Failed to find RX ring for key {}", key);
              params_rx->rxqw[ridx].ring = nullptr;  // Handle error
            } else {
              params_rx->rxqw[ridx].ring = ring_it->second;
            }

            auto qinfo = rx_q_map_[key];
            params_rx->rxqw[ridx].queue = q.common_.id_;
            params_rx->rxqw[ridx].batch_size = q.common_.batch_size_;
            params_rx->rxqw[ridx].rxq = qinfo;
            params_rx->rxqw[ridx].port = intf.port_id_;

            ridx++;
          }
        }
      }

      if (ridx > 0) {
        worker_th[worker_th_idx++] = std::thread(rx_worker, (void*)params_rx);
        ridx = 0;
      }
    }
  }

  if (tx_enabled) {
    for (const auto gpu_idx : gpu_mr_devs) {
      params_tx = new TxDocaWorkerParams;

      params_tx->meta_pool = tx_metadata;
      params_tx->gpu_id = gpu_idx;  // cfg_.mrs_[tx.queues_[0].common_.mrs_[0]].affinity_;
      params_tx->gdev = gdev[params_tx->gpu_id];
      params_tx->txqn = 0;
      for (const auto& intf : cfg_.ifs_) {
        const auto& tx = intf.tx_;

        for (auto& q : tx.queues_) {
          if (cfg_.mrs_[q.common_.mrs_[0]].affinity_ == gpu_idx) {
            params_tx->txqn++;
            if (tidx == 0) {
              params_tx->core_id = stoi(q.common_.cpu_core_);
              rte_eth_macaddr_get(intf.port_id_, &params_tx->mac_addr);
            }

            uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
            auto qinfo = tx_q_map_[key];
            params_tx->txqw[tidx].queue = q.common_.id_;
            params_tx->txqw[tidx].batch_size = q.common_.batch_size_;
            params_tx->txqw[tidx].txq = qinfo;
            params_tx->txqw[tidx].port = intf.port_id_;
            params_tx->txqw[tidx].ring = tx_rings[key];
            tidx++;
          }
        }
      }

      if (tidx > 0) {
        worker_th[worker_th_idx++] = std::thread(tx_worker, (void*)params_tx);
        tidx = 0;
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
  uint32_t *sem_idx_cpu_list, *sem_idx_gpu_list;
  uint32_t *batch_cpu_list, *batch_gpu_list;
  uint32_t *cpu_exit_condition, *gpu_exit_condition;
  // int sem_idx[MAX_NUM_RX_QUEUES] = {0};
  struct adv_doca_rx_gpu_info* packets_stats;
  BurstParams* burst;
#if MPS_ENABLED == 1
  CUdevice cuDevice;
  CUcontext cuContext;
#endif
  uint64_t last_batch = 0;
  int leastPriority;
  int greatestPriority;

  pthread_t self = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(tparams->core_id, &cpuset);

  int rc = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    printf("Failed to pin core: %s\n", strerror(errno));
    exit(1);
  }
  pthread_setname_np(self, "RX_WORKER");

  // WAR for Holoscan management of threads.
  // Be sure application thread finished before launching other CUDA tasks
  sleep(2);

  // Log
  {
    std::string queue_ids_str = "";
    for (int idxq = 0; idxq < tparams->rxqn; ++idxq) {
      queue_ids_str += std::to_string(tparams->rxqw[idxq].queue);
      if (idxq < tparams->rxqn - 1) {
        queue_ids_str += ", ";
      }
    }
    if (tparams->rxqn == 0) { queue_ids_str = "none"; }
    HOLOSCAN_LOG_INFO("Starting Rx Core {} (queue IDs: {}), GPU {}",
                      tparams->core_id,
                      queue_ids_str,
                      tparams->gpu_id);
  }

  cudaSetDevice(tparams->gpu_id);
  // cudaFree(0);
#if MPS_ENABLED == 1
  cuDeviceGet(&cuDevice, tparams->gpu_id);
  cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
  cuCtxPushCurrent(cuContext);
#endif
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uintptr_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&eth_rxq_gpu_list,
                              (void**)&eth_rxq_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "Failed to allocate gpu memory eth_rxq_gpu_list before launching kernel {} "
        "Core {} ({} queues), GPU {}, socket {}",
        doca_error_get_descr(result),
        tparams->core_id,
        tparams->rxqn,
        tparams->gpu_id,
        rte_socket_id());
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uintptr_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&sem_gpu_list,
                              (void**)&sem_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory sem_gpu_list before launching kernel {}",
                       doca_error_get_descr(result));
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              tparams->rxqn * sizeof(uint32_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_CPU_GPU,
                              (void**)&sem_idx_gpu_list,
                              (void**)&sem_idx_cpu_list);
  if (result != DOCA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory sem_gpu_list before launching kernel {}",
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
    HOLOSCAN_LOG_ERROR("Failed to allocate gpu memory batch_gpu_list before launching kernel {}",
                       doca_error_get_descr(result));
    exit(1);
  }

  for (int idx = 0; idx < tparams->rxqn; idx++) {
    eth_rxq_cpu_list[idx] = (uintptr_t)tparams->rxqw[idx].rxq->eth_rxq_gpu;
    sem_cpu_list[idx] = (uintptr_t)tparams->rxqw[idx].rxq->sem_gpu;
    sem_idx_cpu_list[idx] = 0;
    batch_cpu_list[idx] = tparams->rxqw[idx].batch_size;
  }

  res_cuda = cudaStreamCreateWithPriority(&rx_stream, cudaStreamNonBlocking, greatestPriority);
  if (res_cuda != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Function cudaStreamCreateWithPriority error {}",
                       static_cast<int>(res_cuda));
    exit(1);
  }

  result = doca_gpu_mem_alloc(tparams->gdev,
                              GPU_PAGE_SIZE,  // sizeof(uint32_t),
                              GPU_PAGE_SIZE,
                              DOCA_GPU_MEM_TYPE_GPU_CPU,
                              (void**)&gpu_exit_condition,
                              (void**)&cpu_exit_condition);
  if (result != DOCA_SUCCESS || gpu_exit_condition == nullptr || cpu_exit_condition == nullptr) {
    HOLOSCAN_LOG_ERROR("Function doca_gpu_mem_alloc returned {}", doca_error_get_descr(result));
    exit(1);
  }
  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

#if ADV_NETWORK_MANAGER_WARMUP_KERNEL
  HOLOSCAN_LOG_INFO("Warmup receive kernel");
  doca_receiver_packet_kernel(rx_stream,
                              tparams->rxqn,
                              nullptr,
                              sem_gpu_list,
                              sem_idx_gpu_list,
                              batch_gpu_list,
                              gpu_exit_condition,
                              false);
#endif
  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
  cudaStreamSynchronize(rx_stream);

  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

  doca_receiver_packet_kernel(rx_stream,
                              tparams->rxqn,
                              eth_rxq_gpu_list,
                              sem_gpu_list,
                              sem_idx_gpu_list,
                              batch_gpu_list,
                              gpu_exit_condition,
                              true);

  HOLOSCAN_LOG_INFO("DOCA receiver kernel ready!");

  uint64_t loop_count = 0;
  uint64_t loop_log_rate = 100000000;
  while (!force_quit_doca.load()) {
    loop_count++;

    for (int ridx = 0; ridx < tparams->rxqn; ridx++) {
      result = doca_gpu_semaphore_get_status(
          tparams->rxqw[ridx].rxq->sem_cpu, sem_idx_cpu_list[ridx], &status);
      if (result != DOCA_SUCCESS) {
        HOLOSCAN_LOG_ERROR("UDP semaphore error queue {}, result={}.", ridx, (int)result);
        force_quit_doca.store(true);
        break;
      }

      // Log semaphore status periodically unless it's ready
      if (status != DOCA_GPU_SEMAPHORE_STATUS_READY && (loop_count % loop_log_rate == 0)) {
        HOLOSCAN_LOG_INFO(
            "rx_core Q {}, sem_idx {}, status: {}", ridx, sem_idx_cpu_list[ridx], (int)status);
      }

      if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
        result = doca_gpu_semaphore_get_custom_info_addr(
            tparams->rxqw[ridx].rxq->sem_cpu, sem_idx_cpu_list[ridx], (void**)&(packets_stats));
        if (result != DOCA_SUCCESS) {
          HOLOSCAN_LOG_ERROR("UDP semaphore get address error.");
          force_quit_doca.store(true);
          break;
        }

        if (rte_mempool_get(tparams->meta_pool, reinterpret_cast<void**>(&burst)) < 0) {
          HOLOSCAN_LOG_ERROR("Processing function falling behind. No free buffers for metadata!");
          force_quit_doca.store(true);
          break;
        }

        //  Queue ID for receiver to differentiate
        burst->hdr.hdr.q_id = tparams->rxqw[ridx].queue;
        burst->hdr.hdr.first_pkt_addr = (uintptr_t)tparams->rxqw[ridx].rxq->gpu_pkt_addr;
        burst->hdr.hdr.max_pkt = tparams->rxqw[ridx].rxq->max_pkt_num;
        burst->hdr.hdr.max_pkt_size = tparams->rxqw[ridx].rxq->max_pkt_size;
        burst->hdr.hdr.port_id = tparams->rxqw[ridx].port;
        burst->hdr.hdr.num_pkts = packets_stats->num_pkts;
        burst->hdr.hdr.nbytes = packets_stats->nbytes;
        burst->hdr.hdr.gpu_pkt0_idx = packets_stats->gpu_pkt0_idx;
        burst->hdr.hdr.gpu_pkt0_addr = packets_stats->gpu_pkt0_addr;
        HOLOSCAN_LOG_DEBUG(
            "sem {} queue {} num_pkts {}", sem_idx_cpu_list[ridx], ridx, burst->hdr.hdr.num_pkts);
        // Check if the ring pointer assigned during setup is valid
        if (tparams->rxqw[ridx].ring == nullptr) {
          HOLOSCAN_LOG_ERROR("RX Worker: Ring pointer for queue index {} is null. Dropping burst.",
                             ridx);
          rte_mempool_put(tparams->meta_pool, burst);
        } else {
          // Enqueue into the specific ring associated with this worker queue
          if (rte_ring_enqueue(tparams->rxqw[ridx].ring, reinterpret_cast<void*>(burst)) != 0) {
            HOLOSCAN_LOG_WARN("RX ring for queue index {} is full. Dropping burst.", ridx);
            rte_mempool_put(tparams->meta_pool, burst);
          }
        }

        // Update stats
        total_pkts += burst->hdr.hdr.num_pkts;
        stats_rx_tot_pkts += burst->hdr.hdr.num_pkts;
        stats_rx_tot_bytes += burst->hdr.hdr.nbytes;
        stats_rx_tot_batch++;

        // Reset semaphore to free
        result = doca_gpu_semaphore_set_status(tparams->rxqw[ridx].rxq->sem_cpu,
                                               sem_idx_cpu_list[ridx],
                                               DOCA_GPU_SEMAPHORE_STATUS_FREE);
        if (result != DOCA_SUCCESS) {
          HOLOSCAN_LOG_ERROR("UDP semaphore set status error queue {}.", ridx);
          force_quit_doca.store(true);
          break;
        }

        sem_idx_cpu_list[ridx] = (sem_idx_cpu_list[ridx] + 1) % MAX_DEFAULT_SEM_X_QUEUE;
      }
    }
  }

  DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
  HOLOSCAN_LOG_INFO("Wait receive kernel completion");
  cudaStreamSynchronize(rx_stream);

  for (int ridx = 0; ridx < tparams->rxqn; ridx++) {
    // HOLOSCAN_LOG_INFO("Check queue {} sem {}", ridx, sem_idx[ridx]);
    doca_gpu_semaphore_get_status(
        tparams->rxqw[ridx].rxq->sem_cpu, sem_idx_cpu_list[ridx], &status);
    if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
      doca_gpu_semaphore_get_custom_info_addr(
          tparams->rxqw[ridx].rxq->sem_cpu, sem_idx_cpu_list[ridx], (void**)&(packets_stats));
      last_batch += packets_stats->num_pkts;
      stats_rx_tot_pkts += packets_stats->num_pkts;
      stats_rx_tot_bytes += packets_stats->nbytes;
      stats_rx_tot_batch++;
    }
  }

  doca_gpu_mem_free(tparams->gdev, (void*)eth_rxq_gpu_list);
  doca_gpu_mem_free(tparams->gdev, (void*)sem_gpu_list);
  doca_gpu_mem_free(tparams->gdev, (void*)sem_idx_gpu_list);
  cudaStreamDestroy(rx_stream);
  doca_gpu_mem_free(tparams->gdev, (void*)gpu_exit_condition);

#if MPS_ENABLED == 1
  cuCtxPopCurrent(&cuContext);
#endif

  HOLOSCAN_LOG_INFO(
      "Total packets received by application (GPU {}): {}, last partial batch packets {}",
      tparams->gpu_id,
      total_pkts + last_batch,
      last_batch);

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
  BurstParams* burst;
  uint64_t cnt_pkts[MAX_DEFAULT_QUEUES] = {0};
  bool set_completion[MAX_DEFAULT_QUEUES] = {false};
#if MPS_ENABLED == 1
  CUdevice cuDevice;
  CUcontext cuContext;
#endif
  pthread_t self = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(tparams->core_id, &cpuset);

  int rc = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    printf("Failed to pin core: %s\n", strerror(errno));
    exit(1);
  }
  pthread_setname_np(self, "TX_WORKER");

  // Log
  {
    std::string queue_ids_str = "";
    for (int idxq = 0; idxq < tparams->txqn; ++idxq) {
      queue_ids_str += std::to_string(tparams->txqw[idxq].queue);
      if (idxq < tparams->txqn - 1) {
        queue_ids_str += ", ";
      }
    }
    if (tparams->txqn == 0) { queue_ids_str = "none"; }
    HOLOSCAN_LOG_INFO("Starting Tx Core {} (queue IDs: {}), GPU {}",
                      tparams->core_id,
                      queue_ids_str,
                      tparams->gpu_id);
  }

  cudaSetDevice(tparams->gpu_id);
  // cudaFree(0);
#if MPS_ENABLED == 1
  cuDeviceGet(&cuDevice, tparams->gpu_id);
  cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
  cuCtxPushCurrent(cuContext);
#endif

  for (int idxq = 0; idxq < tparams->txqn; idxq++) {
    res_cuda = cudaStreamCreateWithFlags(&tx_stream[idxq], cudaStreamNonBlocking);
    if (res_cuda != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Function cudaStreamCreateWithFlags error {}", static_cast<int>(res_cuda));
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

      /* Only necessary checks to prioritize the launch of the kernel */
      cnt_pkts[idxq] += burst->hdr.hdr.num_pkts;
      if (cnt_pkts[idxq] > MAX_SQ_DESCR_NUM / 4) set_completion[idxq] = true;

      doca_sender_packet_kernel(tx_stream[idxq],
                                tparams->txqw[idxq].txq->eth_txq_gpu,
                                tparams->txqw[idxq].txq->buf_arr_gpu,
                                burst->hdr.hdr.gpu_pkt0_idx,
                                burst->hdr.hdr.num_pkts,
                                burst->hdr.hdr.max_pkt,
                                burst->pkt_lens[0],
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

  HOLOSCAN_LOG_DEBUG("DOCA RX must exit");

  for (int idxq = 0; idxq < tparams->txqn; idxq++) {
    res_cuda = cudaStreamDestroy(tx_stream[idxq]);
    if (res_cuda != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Function cudaStreamDestroy error {}", static_cast<int>(res_cuda));
    }
  }

#if MPS_ENABLED == 1
  cuCtxPopCurrent(&cuContext);
#endif

  return 0;
}

/* advanced_network interface implementations */

void* DocaMgr::get_packet_ptr(BurstParams* burst, int idx) {
  uint32_t pkt = burst->hdr.hdr.gpu_pkt0_idx + idx;

  // HOLOSCAN_LOG_INFO("get_gpu_pkt_ptr pkt {} gpu_pkt0_idx {} idx {} addr {}\n",
  //         pkt, burst->hdr.hdr.gpu_pkt0_idx, idx, burst->hdr.hdr.gpu_pkt0_addr);

  if (pkt < burst->hdr.hdr.max_pkt)
    return (void*)(((uintptr_t)burst->hdr.hdr.gpu_pkt0_addr) + (idx * burst->hdr.hdr.max_pkt_size));
  else
    return (void*)(((uintptr_t)burst->hdr.hdr.first_pkt_addr) +
                   ((pkt % burst->hdr.hdr.max_pkt) * burst->hdr.hdr.max_pkt_size));
}

void* DocaMgr::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
  if (seg > 0) {
    HOLOSCAN_LOG_CRITICAL("DOCA GPU comms doesn't support multiple segments yet!");
    return nullptr;
  }

  return get_packet_ptr(burst, idx);
}

void* DocaMgr::get_packet_extra_info(BurstParams* burst, int idx) {
  return nullptr;
}

uint64_t DocaMgr::get_burst_tot_byte(BurstParams* burst) {
  return burst->hdr.hdr.nbytes;
}

uint32_t DocaMgr::get_packet_length(BurstParams* burst, int idx) {
  return 0;
}

uint16_t DocaMgr::get_packet_flow_id(BurstParams* burst, int idx) {
  return 0;
}

uint32_t DocaMgr::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
  return 0;
}

Status DocaMgr::get_mac_addr(int port, char* mac) {
  if (port > 0) {
    HOLOSCAN_LOG_CRITICAL("Port {} out of range in get_mac_addr() lookup");
    return Status::INVALID_PARAMETER;
  }

  memcpy(mac, reinterpret_cast<char*>(&mac_addrs[port]), sizeof(mac_addrs[port]));
  return Status::SUCCESS;
}

Status DocaMgr::set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp) {
  return Status::SUCCESS;
}

Status DocaMgr::get_tx_packet_burst(BurstParams* burst) {
  int buf_idx = 0;

  // Check if burst->hdr.hdr.num_pkts > max_tx_batch_size

  for (const auto& intf : cfg_.ifs_) {
    if (burst->hdr.hdr.port_id != intf.port_id_) { continue; }

    for (auto& q : intf.tx_.queues_) {
      if (q.common_.id_ == burst->hdr.hdr.q_id) {
        uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
        auto txq = tx_q_map_[key];

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
            intf.port_id_,
            burst->hdr.hdr.num_pkts,
            burst->hdr.hdr.first_pkt_addr,
            burst->hdr.hdr.gpu_pkt0_idx);
      }
    }
  }

  return Status::SUCCESS;
}

Status DocaMgr::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  return Status::NOT_SUPPORTED;
}

Status DocaMgr::set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                unsigned int src_host, unsigned int dst_host) {
  return Status::NOT_SUPPORTED;
}

Status DocaMgr::set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                               uint16_t dst_port) {
  return Status::NOT_SUPPORTED;
}

Status DocaMgr::set_udp_payload(BurstParams* burst, int idx, void* data, int len) {
  return Status::NOT_SUPPORTED;
}

bool DocaMgr::is_tx_burst_available(BurstParams* burst) {
  for (const auto& intf : cfg_.ifs_) {
    if (burst->hdr.hdr.port_id != intf.port_id_) { continue; }

    for (auto& q : intf.tx_.queues_) {
      if (q.common_.id_ == burst->hdr.hdr.q_id) {
        uint32_t key = generate_queue_key(intf.port_id_, q.common_.id_);
        auto txq = tx_q_map_[key];
        doca_pe_progress(txq->pe);
        if (txq->tx_cmp_posted > TX_COMP_THRS) {
          HOLOSCAN_LOG_DEBUG("txq->tx_cmp_posted {}", static_cast<int>(txq->tx_cmp_posted.load()));
          return false;
        }

        return true;
      }
    }
  }

  return true;
}

Status DocaMgr::set_packet_lengths(BurstParams* burst, int idx,
                                   const std::initializer_list<int>& lens) {
  burst->pkt_lens[0][idx] = *(lens.begin());
  return Status::SUCCESS;
}

void DocaMgr::free_rx_burst(BurstParams* burst) {
  rte_mempool_put(rx_metadata, burst);
}

void DocaMgr::free_tx_burst(BurstParams* burst) {
  return;
}

Status DocaMgr::get_rx_burst(BurstParams** burst, int port, int q) {
  uint32_t key = generate_queue_key(port, q);
  auto ring_it = rx_rings.find(key);

  if (ring_it == rx_rings.end()) {
    HOLOSCAN_LOG_ERROR("get_rx_burst: Could not find ring for port {}, queue {}. Check config.",
                       port,
                       q);
    return Status::INVALID_PARAMETER;
  }

  if (rte_ring_dequeue(ring_it->second, reinterpret_cast<void**>(burst)) < 0) {
    return Status::NULL_PTR;  // Use NULL_PTR to indicate empty queue
  }

  return Status::SUCCESS;
}

void DocaMgr::free_rx_metadata(BurstParams* burst) {
  rte_mempool_put(rx_metadata, burst);
}

void DocaMgr::free_tx_metadata(BurstParams* burst) {
  rte_mempool_put(tx_metadata, burst);
}

BurstParams* DocaMgr::create_tx_burst_params() {
  auto burst_idx = burst_tx_idx.fetch_add(1);
  HOLOSCAN_LOG_DEBUG("create_tx_burst_params burst_idx {} MAX_TX_BURST {}",
                     burst_idx % MAX_TX_BURST,
                     MAX_TX_BURST);
  return &(burst[burst_idx % MAX_TX_BURST]);
}

Status DocaMgr::get_tx_metadata_buffer(BurstParams** burst) {
  if (rte_mempool_get(tx_metadata, reinterpret_cast<void**>(burst)) != 0) {
    fprintf(stderr, "Failed to get TX meta descriptor\n");
    HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor");
    return Status::NO_FREE_BURST_BUFFERS;
  }

  return Status::SUCCESS;
}

Status DocaMgr::send_tx_burst(BurstParams* burst) {
  uint32_t key = generate_queue_key(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
  const auto ring = tx_rings.find(key);

  if (ring == tx_rings.end()) {
    HOLOSCAN_LOG_ERROR("Invalid port/queue combination in send_tx_burst: {}/{}",
                       burst->hdr.hdr.port_id,
                       burst->hdr.hdr.q_id);
    return Status::INVALID_PARAMETER;
  }

  if (rte_ring_enqueue(ring->second, reinterpret_cast<void*>(burst)) != 0) {
    fprintf(stderr, "calling DOCA free_tx_metadata\n");
    free_tx_metadata(burst);
    HOLOSCAN_LOG_CRITICAL("Failed to enqueue TX work");
    return Status::NO_SPACE_AVAILABLE;
  }

  return Status::SUCCESS;
}

void DocaMgr::shutdown() {
  int icore = 0;

  HOLOSCAN_LOG_INFO("advanced_network DOCA manager shutting down");

  if (force_quit_doca.load() == false) {
    print_stats();

    HOLOSCAN_LOG_INFO("advanced_network DOCA manager stopping cores");
    force_quit_doca.store(true);
    for (int i = 0; i < worker_th_idx; i++) {
      HOLOSCAN_LOG_INFO("Waiting on thread {}", i);
      worker_th[i].join();
    }
  }

  // RTE_LCORE_FOREACH_WORKER(icore) {
  //   if (rte_eal_wait_lcore(icore) < 0) {
  //     fprintf(stderr, "bad exit for coreid: %d\n", icore);
  //     break;
  //   }
  // }
}

void DocaMgr::print_stats() {
  HOLOSCAN_LOG_INFO("advanced_network DOCA manager stats");
  HOLOSCAN_LOG_INFO("Total Rx packets {}", stats_rx_tot_pkts);
  HOLOSCAN_LOG_INFO("Total Rx bytes {}", stats_rx_tot_bytes);
  HOLOSCAN_LOG_INFO("Total Rx batch processed {}", stats_rx_tot_batch);

  HOLOSCAN_LOG_INFO("Total Tx packets {}", stats_tx_tot_pkts);
  HOLOSCAN_LOG_INFO("Total Tx bytes {}", stats_tx_tot_bytes);
  HOLOSCAN_LOG_INFO("Total Tx batch processed {}", stats_tx_tot_batch);

  return;
}

};  // namespace holoscan::advanced_network
