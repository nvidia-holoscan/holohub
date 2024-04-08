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
#include <memory>
#include <optional>
#include <stdint.h>
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

/**
 * @brief Reserved header bytes for burst structure
 *
 */
static inline constexpr uint32_t ADV_NETWORK_HEADER_SIZE_BYTES = 256;
static inline constexpr uint32_t MAX_NUM_RX_QUEUES = 32;
static inline constexpr uint32_t MAX_NUM_TX_QUEUES = 32;
static inline constexpr uint32_t MAX_INTERFACES = 4;

/**
 * @brief Header of AdvNetBurstParams
 *
 */
struct AdvNetBurstHdrParams {
  size_t        num_pkts;
  uint16_t      port_id;
  uint16_t      q_id;
};

struct AdvNetBurstHdr {
    AdvNetBurstHdrParams hdr;

    // Pad without union to make bindings readable
    uint8_t pad[ADV_NETWORK_HEADER_SIZE_BYTES - sizeof(AdvNetBurstHdrParams)];
};

struct AdvNetBurstParams {
  AdvNetBurstHdr hdr;

  void **cpu_pkts;
  void **gpu_pkts;
};



/**
 * @brief Return status codes from advanced network operators
 *
 */
enum class AdvNetStatus {
  SUCCESS,
  NULL_PTR,
  NO_FREE_BURST_BUFFERS,
  NO_FREE_CPU_PACKET_BUFFERS,
  NO_FREE_GPU_PACKET_BUFFERS,
  NOT_READY,
  INVALID_PARAMETER,
  NO_SPACE_AVAILABLE
};

/**
 * @brief Location of packet buffers
 *
 */
enum class AdvNetBufferLocation : uint8_t {
  CPU = 0,
  GPU = 1,
  CPU_GPU_SPLIT = 2,
};

/**
 * @brief Direction of operator
 *
 */
enum class AdvNetDirection : uint8_t {
  RX = 0,
  TX = 1,
  TX_RX = 2,
};

/**
 * @brief Parameters to configure advanced network operators
 *
 */
struct AdvNetConfig {
  AdvNetBufferLocation rx_loc;
  uint16_t max_packet_size = 0;
  size_t batch_size = 0;
  uint32_t num_concurrent_batches;
  std::vector<std::string> if_names;
  std::string cpu_cores = "";
  std::string master_core = "";
  int hds = 0;
  int gpu_dev = -1;
  bool use_network = false;
  bool enabled = false;
};


struct CommonQueueConfig {
  std::string name_;
  int id_;
  int gpu_dev_;
  bool gpu_direct_;
  int hds_;
  std::string cpu_cores_;
  int max_packet_size_;
  int num_concurrent_batches_;
  int batch_size_;
  void *backend_config_;  // Filled in by operator
};

struct RxQueueConfig {
  std::string output_port_;
  CommonQueueConfig common_;
};

struct TxQueueConfig {
  CommonQueueConfig common_;
};

// struct FlowConfig {
//   FlowConfig() = default;
//   std::string name_;
//   std::string pattern_;
// };

enum class FlowType {
  QUEUE
};

struct FlowAction {
  FlowType type_;
  uint16_t id_;
};

struct FlowMatch {
  uint16_t udp_src_;
  uint16_t udp_dst_;
};
struct FlowConfig {
  std::string name_;
  FlowAction action_;
  FlowMatch  match_;
};

struct CommonConfig {
  int version;
  int master_core_;
  std::string mgr_;
  AdvNetDirection dir;
};

struct AdvNetRxConfig {
    std::string if_name_;
    uint16_t port_id_;
    bool flow_isolation_;
    bool empty;
    std::vector<RxQueueConfig> queues_;
    std::vector<FlowConfig> flows_;
};

struct AdvNetTxConfig {
    std::string if_name_;
    bool accurate_send_;
    uint16_t port_id_;
    bool empty;
    std::vector<TxQueueConfig> queues_;
    std::vector<FlowConfig> flows_;
};

struct AdvNetConfigYaml {
    CommonConfig common_;
    std::vector<AdvNetRxConfig> rx_;
    std::vector<AdvNetTxConfig> tx_;
};

template <typename Config>
auto adv_net_get_rx_tx_cfg_en(const Config &config) {
  bool rx = false;
  bool tx = false;

  auto& yaml_nodes = config.yaml_nodes();
  for (const auto &yaml_node : yaml_nodes) {
    auto node = yaml_node["advanced_network"]["cfg"];
    rx = node["rx"].IsSequence();
    tx = node["tx"].IsSequence();
  }

  return std::make_tuple(rx, tx);
}

template <typename Config>
std::string adv_net_get_manager(const Config &config) {
  auto& yaml_nodes = config.yaml_nodes();
  for (const auto &yaml_node : yaml_nodes) {
    try {
      auto node = yaml_node["advanced_network"]["cfg"];
      return node["manager"].template as<std::string>();
    }
    catch (const std::exception& e) {
      return "default";
    }
  }

  return "default";
}

};  // namespace holoscan::ops
