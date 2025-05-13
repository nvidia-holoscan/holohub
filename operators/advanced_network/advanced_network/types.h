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

namespace holoscan::advanced_network {

/**
 * @brief Reserved header bytes for burst structure
 *
 */
static inline constexpr uint32_t ADV_NETWORK_HEADER_SIZE_BYTES = 256;
static inline constexpr uint32_t MAX_NUM_RX_QUEUES = 32;
static inline constexpr uint32_t MAX_NUM_TX_QUEUES = 32;
static inline constexpr uint32_t MAX_INTERFACES = 4;
static inline constexpr int MAX_NUM_SEGS = 4;

/**
 * @brief Return status codes from communication with the NIC
 *
 */
enum class Status {
  SUCCESS,
  NULL_PTR,
  NO_FREE_BURST_BUFFERS,
  NO_FREE_PACKET_BUFFERS,
  NOT_READY,
  INVALID_PARAMETER,
  NO_SPACE_AVAILABLE,
  NOT_SUPPORTED,
  GENERIC_FAILURE,
  CONNECT_FAILURE,
  INTERNAL_ERROR,
};

enum class RDMAOpCode {
  CONNECT,
  SEND,
  RECEIVE,
  RDMA_WRITE,
  RDMA_WRITE_IMM,
  RDMA_READ,
  RDMA_READ_IMM,
  INVALID
};

enum class RDMACompletionType { RX, TX, INVALID };

struct AdvNetRdmaBurstHdr {
  uint8_t version;
  RDMAOpCode opcode;
  Status status;
  uint16_t port_id;
  uint16_t q_id;
  bool server;
  bool tx;
  size_t num_pkts;
  int num_segs;
  uint64_t wr_id;
  uintptr_t conn_id;
  char local_mr_name[32];
  char remote_mr_name[32];
  void* raddr;
  uint64_t dst_key;
  uint32_t imm;
  uint32_t server_addr;
  uint16_t server_port;
};

/**
 * @brief Header of BurstParams
 *
 */
struct BurstHeaderParams {
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

struct BurstHeader {
  BurstHeaderParams hdr;

  // Pad without union to make bindings readable
  void* extra_burst_data;
  uint8_t
      custom_burst_data[ADV_NETWORK_HEADER_SIZE_BYTES - sizeof(void*) - sizeof(BurstHeaderParams)];
};

/**
 * @brief Structure for passing packets
 *
 * The BurstParams structure describes metadata about a packet batch and its packet pointers.
 *
 */
struct BurstParams {
  union {
    BurstHeader hdr;
    AdvNetRdmaBurstHdr rdma_hdr;
  };

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
  uint32_t access = 0;
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
 * @brief Location of packet buffers
 *
 */
enum class BufferLocation : uint8_t {
  CPU = 0,
  GPU = 1,
  CPU_GPU_SPLIT = 2,
};

/**
 * @brief Direction of operator
 *
 */
enum class Direction : uint8_t {
  RX = 0,
  TX = 1,
  TX_RX = 2,
};

/**
 * @brief Manager Type
 *
 */
enum class ManagerType {
  UNKNOWN = -1,
  DEFAULT,
  DPDK,
  DOCA,
  RIVERMAX,
  RDMA,
};

static constexpr const char* ANO_MGR_STR__DPDK = "dpdk";
static constexpr const char* ANO_MGR_STR__GPUNETIO = "gpunetio";
static constexpr const char* ANO_MGR_STR__RIVERMAX = "rivermax";
static constexpr const char* ANO_MGR_STR__RDMA = "rdma";
static constexpr const char* ANO_MGR_STR__DEFAULT = "default";
/**
 * @brief Convert string to manager type
 *
 * @param str
 * @return ManagerType
 */
inline ManagerType manager_type_from_string(const std::string& str) {
  if (str == ANO_MGR_STR__DPDK) return ManagerType::DPDK;
  if (str == ANO_MGR_STR__GPUNETIO) return ManagerType::DOCA;
  if (str == ANO_MGR_STR__RIVERMAX) return ManagerType::RIVERMAX;
  if (str == ANO_MGR_STR__RDMA) return ManagerType::RDMA;
  if (str == ANO_MGR_STR__DEFAULT) return ManagerType::DEFAULT;
  throw std::logic_error(std::string("Unknown manager type. Valid options: ") + ANO_MGR_STR__DPDK +
                         "/" + ANO_MGR_STR__GPUNETIO + "/" + ANO_MGR_STR__RIVERMAX + "/" +
                         ANO_MGR_STR__RDMA + "/" + ANO_MGR_STR__DEFAULT);
}

enum class RDMAMode {
  CLIENT,
  SERVER,

  INVALID
};

inline RDMAMode GetRDMAModeFromString(const std::string& mode_str) {
  if (mode_str == "client") {
    return RDMAMode::CLIENT;
  } else if (mode_str == "server") {
    return RDMAMode::SERVER;
  }

  return RDMAMode::INVALID;
}

enum class RDMATransportMode {
  RC,
  UC,
  UD,

  INVALID
};

inline RDMATransportMode GetRDMATransportModeFromString(const std::string& mode_str) {
  if (mode_str == "RC") {
    return RDMATransportMode::RC;
  } else if (mode_str == "UC") {
    return RDMATransportMode::UC;
  }

  return RDMATransportMode::INVALID;
}

struct RDMAConfig {
  RDMAMode mode_ = RDMAMode::INVALID;
  RDMATransportMode xmode_ = RDMATransportMode::INVALID;
  uint16_t port_ = 0;
};

/**
 * @brief Convert manager type to string
 *
 * @param type
 * @return std::string
 */
inline std::string manager_type_to_string(ManagerType type) {
  switch (type) {
    case ManagerType::DPDK:
      return ANO_MGR_STR__DPDK;
    case ManagerType::DOCA:
      return ANO_MGR_STR__GPUNETIO;
    case ManagerType::RIVERMAX:
      return ANO_MGR_STR__RIVERMAX;
    case ManagerType::RDMA:
      return ANO_MGR_STR__RDMA;
    case ManagerType::DEFAULT:
      return ANO_MGR_STR__DEFAULT;
    default:
      return "unknown";
  }
}
class LogLevel {
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
 * @brief Base class for additional queue configuration.
 *
 * This class serves as a base class for any additional queue configuration
 * that might be needed by different manager types. This class should be
 * inherited by the derived class that will hold the additional configuration
 * for a specific manager type.
 */
class ManagerExtraQueueConfig {
 public:
  /**
   * @brief Virtual destructor for proper cleanup of derived class objects.
   */
  virtual ~ManagerExtraQueueConfig() = default;
};

struct CommonQueueConfig {
  std::string name_;
  int id_;
  int batch_size_;
  int split_boundary_;
  std::string cpu_core_;
  std::vector<std::string> mrs_;
  std::vector<std::string> offloads_;
  ManagerExtraQueueConfig* extra_queue_config_;
};

struct MemoryRegionConfig {
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
  Direction dir;
  ManagerType manager_type;
};

struct RxConfig {
  bool flow_isolation_;
  std::vector<RxQueueConfig> queues_;
  std::vector<FlowConfig> flows_;
};

struct TxConfig {
  bool accurate_send_ = false;
  std::vector<TxQueueConfig> queues_;
  std::vector<FlowConfig> flows_;
};

struct InterfaceConfig {
  std::string name_;
  std::string address_;
  uint16_t port_id_;
  RDMAConfig rdma_;
  RxConfig rx_;
  TxConfig tx_;
};

struct NetworkConfig {
  CommonConfig common_;
  std::unordered_map<std::string, MemoryRegionConfig> mrs_;
  std::vector<InterfaceConfig> ifs_;
  uint16_t debug_;
  LogLevel::Level log_level_;
};

template <typename Config>
auto get_rdma_cfg_en(const Config& config) {
  bool server = false;
  bool client = false;

  auto& yaml_nodes = config.yaml_nodes();
  for (const auto& yaml_node : yaml_nodes) {
    auto node = yaml_node["advanced_network"]["cfg"]["interfaces"];
    for (const auto& intf : node) {
      std::string mode = intf["rdma_mode"].template as<std::string>();
      if (mode == "server") {
        server = true;
      } else if (mode == "client") {
        client = true;
      }
    }
  }

  return std::make_tuple(server, client);
}

template <typename Config>
auto get_rx_tx_configs_enabled(const Config& config) {
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

};  // namespace holoscan::advanced_network
