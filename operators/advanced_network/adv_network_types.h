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
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <linux/if_ether.h>
#include <netinet/ip.h>
#include <linux/udp.h>
#include <cuda_runtime.h>

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
  size_t num_pkts;
  uint16_t port_id;
  uint16_t q_id;
  int num_segs;
  uint32_t nbytes;
  uintptr_t first_pkt_addr;
  uint32_t max_pkt;
  uint32_t max_pkt_size;
  uint32_t gpu_pkt0_idx;
  uintptr_t gpu_pkt0_addr;
};

struct AdvNetBurstHdr {
  AdvNetBurstHdrParams hdr;

  // Pad without union to make bindings readable
  void* extra_burst_data;
  uint8_t custom_burst_data[ADV_NETWORK_HEADER_SIZE_BYTES - sizeof(void*) -
                            sizeof(AdvNetBurstHdrParams)];
};

static inline constexpr int MAX_NUM_SEGS = 4;
struct AdvNetBurstParams {
  AdvNetBurstHdr hdr;

  std::array<void**, MAX_NUM_SEGS> pkts;
  std::array<uint32_t*, MAX_NUM_SEGS> pkt_lens;
  void** pkt_extra_info;
  cudaEvent_t event;
};

// Example IPV4 UDP packet using Linux headers
struct UDPIPV4Pkt {
  struct ethhdr eth;
  struct iphdr ip;
  struct udphdr udp;
} __attribute__((packed));

enum class MemoryKind { HOST, HOST_PINNED, HUGE, DEVICE, INVALID };

enum MemoryAccess {
  MEM_ACCESS_LOCAL = 1U,
  MEM_ACCESS_RDMA_WRITE = 1U << 1,
  MEM_ACCESS_RDMA_READ = 1U << 2
};

inline MemoryKind GetMemoryKindFromString(const std::string& mode_str) {
  if (mode_str == "host") {
    return MemoryKind::HOST;
  } else if (mode_str == "host_pinned") {
    return MemoryKind::HOST_PINNED;
  } else if (mode_str == "huge") {
    return MemoryKind::HUGE;
  } else if (mode_str == "device") {
    return MemoryKind::DEVICE;
  }

  return MemoryKind::INVALID;
}

template <typename T>
uint32_t GetMemoryAccessPropertiesFromList(const T& list) {
  uint32_t access;
  for (const auto& it : list) {
    const auto str = it.template as<std::string>();
    if (str == "local") {
      access |= MEM_ACCESS_LOCAL;
    } else if (str == "rdma_write") {
      access |= MEM_ACCESS_RDMA_WRITE;
    } else if (str == "rdma_read") {
      access |= MEM_ACCESS_RDMA_WRITE;
    } else {
      return 0;
    }
  }

  return access;
}

/**
 * @brief Return status codes from advanced network operators
 *
 */
enum class AdvNetStatus {
  SUCCESS,
  NULL_PTR,
  NO_FREE_BURST_BUFFERS,
  NO_FREE_PACKET_BUFFERS,
  NOT_READY,
  INVALID_PARAMETER,
  NO_SPACE_AVAILABLE,
  NOT_SUPPORTED
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
 * @brief Manager Type
 *
 */
enum class AnoMgrType {
  UNKNOWN = -1,
  DEFAULT,
  DPDK,
  DOCA,
  RIVERMAX,
};

static constexpr const char* ANO_MGR_STR__DPDK = "dpdk";
static constexpr const char* ANO_MGR_STR__GPUNETIO = "gpunetio";
static constexpr const char* ANO_MGR_STR__RIVERMAX = "rivermax";
static constexpr const char* ANO_MGR_STR__DEFAULT = "default";

/**
 * @brief Convert string to manager type
 *
 * @param str
 * @return AnoMgrType
 */
inline AnoMgrType manager_type_from_string(const std::string& str) {
  if (str == ANO_MGR_STR__DPDK) return AnoMgrType::DPDK;
  if (str == ANO_MGR_STR__GPUNETIO) return AnoMgrType::DOCA;
  if (str == ANO_MGR_STR__RIVERMAX) return AnoMgrType::RIVERMAX;
  if (str == ANO_MGR_STR__DEFAULT) return AnoMgrType::DEFAULT;
  throw std::logic_error(std::string("Unknown manager type. Valid options: ") +
                        ANO_MGR_STR__DPDK + "/" +
                        ANO_MGR_STR__GPUNETIO + "/" +
                        ANO_MGR_STR__RIVERMAX + "/" +
                        ANO_MGR_STR__DEFAULT);
}

/**
 * @brief Convert manager type to string
 *
 * @param type
 * @return std::string
 */
inline std::string manager_type_to_string(AnoMgrType type) {
  switch (type) {
    case AnoMgrType::DPDK:
      return ANO_MGR_STR__DPDK;
    case AnoMgrType::DOCA:
      return ANO_MGR_STR__GPUNETIO;
    case AnoMgrType::RIVERMAX:
      return ANO_MGR_STR__RIVERMAX;
    case AnoMgrType::DEFAULT:
      return ANO_MGR_STR__DEFAULT;
    default:
      return "unknown";
  }
}
class AnoLogLevel {
 public:
  enum Level {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL,
    OFF,
  };

  static std::string to_string(Level level) {
    auto it = level_to_string_map.find(level);
    if (it != level_to_string_map.end()) { return it->second; }
    return "warn";
  }

  static Level from_string(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), [](unsigned char c) {
      return std::tolower(c);
    });

    auto it = string_to_level_map.find(lower_str);
    if (it != string_to_level_map.end()) { return it->second; }
    throw std::logic_error(
        "Unrecognized log level, available options trace/debug/info/warn/error/critical/off");
  }

 private:
  static const std::unordered_map<Level, std::string> level_to_string_map;
  static const std::unordered_map<std::string, Level> string_to_level_map;
};

/**
 * @class ManagerLogLevelCommandBuilder
 * @brief Abstract base class for building manager log level commands.
 *
 * This class defines an interface for building commands that manage log levels.
 * Derived classes must implement the `get_cmd_flags_strings` method to provide
 * the specific command flag strings.
 */
class ManagerLogLevelCommandBuilder {
 public:
  /**
   * @brief Virtual destructor for the ManagerLogLevelCommandBuilder class.
   */
  virtual ~ManagerLogLevelCommandBuilder() = default;

  /**
   * @brief Pure virtual function to get the command flag strings.
   *
   * This function must be implemented by derived classes to return
   * the specific command flag strings for managing log levels.
   *
   * @return A vector of command flag strings.
   */
  virtual std::vector<std::string> get_cmd_flags_strings() const = 0;
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
  std::string cpu_core = "";
  std::string master_core = "";
  int hds = 0;
  int gpu_dev = -1;
  bool use_network = false;
  bool enabled = false;
};

/**
 * @brief Base class for additional queue configuration.
 *
 * This class serves as a base class for any additional queue configuration
 * that might be needed by different manager types. This class should be
 * inherited by the derived class that will hold the additional configuration
 * for a specific manager type.
 */
class AnoMgrExtraQueueConfig {
 public:
  /**
   * @brief Virtual destructor for proper cleanup of derived class objects.
   */
  virtual ~AnoMgrExtraQueueConfig() = default;
};

struct CommonQueueConfig {
  std::string name_;
  int id_;
  int batch_size_;
  int split_boundary_;
  std::string cpu_core_;
  std::vector<std::string> mrs_;
  std::vector<std::string> offloads_;
  AnoMgrExtraQueueConfig* extra_queue_config_;
};

struct MemoryRegion {
  std::string name_;
  MemoryKind kind_;
  uint16_t affinity_;
  uint32_t access_;
  size_t buf_size_;
  size_t adj_size_;  // Populated by driver
  size_t ttl_size_;  // Populated by driver
  size_t num_bufs_;
  bool owned_;
};

struct RxQueueConfig {
  CommonQueueConfig common_;
  uint64_t timeout_us_;
  std::string output_port_;
};

struct TxQueueConfig {
  CommonQueueConfig common_;
};

// struct FlowConfig {
//   FlowConfig() = default;
//   std::string name_;
//   std::string pattern_;
// };

enum class FlowType { QUEUE };

struct FlowAction {
  FlowType type_;
  uint16_t id_;
};

struct FlowMatch {
  uint16_t udp_src_;
  uint16_t udp_dst_;
  uint16_t ipv4_len_;
};
struct FlowConfig {
  std::string name_;
  uint16_t id_;
  FlowAction action_;
  FlowMatch match_;
  void* backend_config_;  // Filled in by operator
};

struct CommonConfig {
  int version;
  int master_core_;
  AdvNetDirection dir;
  AnoMgrType manager_type;
};

struct AdvNetRxConfig {
  bool flow_isolation_;
  std::vector<RxQueueConfig> queues_;
  std::vector<FlowConfig> flows_;
};

struct AdvNetTxConfig {
  bool accurate_send_ = false;
  std::vector<TxQueueConfig> queues_;
  std::vector<FlowConfig> flows_;
};

struct AdvNetConfigInterface {
  std::string name_;
  std::string address_;
  uint16_t port_id_;
  AdvNetRxConfig rx_;
  AdvNetTxConfig tx_;
};

struct AdvNetConfigYaml {
  CommonConfig common_;
  std::unordered_map<std::string, MemoryRegion> mrs_;
  std::vector<AdvNetConfigInterface> ifs_;
  uint16_t debug_;
  AnoLogLevel::Level log_level_;
};

template <typename Config>
auto adv_net_get_rx_tx_cfg_en(const Config& config) {
  bool rx = false;
  bool tx = false;

  auto& yaml_nodes = config.yaml_nodes();
  for (const auto& yaml_node : yaml_nodes) {
    auto node = yaml_node["advanced_network"]["cfg"]["interfaces"];
    for (const auto& intf : node) {
      if (intf["rx"]) { rx = true; }
      if (intf["tx"]) { tx = true; }
    }
  }

  return std::make_tuple(rx, tx);
}

};  // namespace holoscan::ops
