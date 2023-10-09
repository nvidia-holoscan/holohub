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
#include <tuple>
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

// this part is purely optional, just a helper for the user
AdvNetBurstParams* adv_net_create_burst_params();


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

namespace detail {
  inline AdvNetDirection DirectionStringToType(const std::string &dir) {
    if (dir == "rx") {
      return AdvNetDirection::RX;
    } else if (dir == "tx") {
      return AdvNetDirection::TX;
    }

    return AdvNetDirection::TX_RX;
  }
};  // namespace detail



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


/**
 * @brief Determine which directions are enabled
 *
 * @param dir Direction from config. Either "rx", "tx", or "tx/rx"
 * @return int Number of directions enabled
 */
inline int EnabledDirections(const std::string &dir) {
  if (dir == "rx" || dir == "tx") {
    return 1;
  }

  return 0;
}


/**
 * @brief Returns a raw CPU packet pointer from a pointer in AdvNetBurstParams
 *
 * The AdvNetBurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the CPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void *adv_net_get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx);
void *adv_net_get_cpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx);

/**
 * @brief Returns a raw GPU packet pointer from a pointer in AdvNetBurstParams
 *
 * The AdvNetBurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the GPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void *adv_net_get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx);
void *adv_net_get_gpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx);


/**
 * @brief Get packet length of a CPU packet
 *
 * Returns the length of an individual CPU packet
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint16_t adv_net_get_cpu_packet_len(AdvNetBurstParams *burst, int idx);
uint16_t adv_net_get_cpu_packet_len(std::shared_ptr<AdvNetBurstParams> burst, int idx);

/**
 * @brief Get packet length of a GPU packet
 *
 * Returns the length of an individual GPU packet
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint16_t adv_net_get_gpu_packet_len(AdvNetBurstParams *burst, int idx);
uint16_t adv_net_get_gpu_packet_len(std::shared_ptr<AdvNetBurstParams> burst, int idx);

/**
 * @brief Populate a TX packet burst buffer
 *
 * Populates a transmit packet burst buffer with allocated packets. The user can take these
 * allocated packets and fill with the desired data/headers.
 *
 * @param burst Burst structure to populate
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packets allocated
 *    NULL_PTR: Burst or packet pools uninitialized
 *    NO_FREE_BURST_BUFFERS: No burst buffers to allocate
 *    NO_FREE_CPU_PACKET_BUFFERS: Not enough CPU packet buffers available
 */
AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams *burst);
AdvNetStatus adv_net_get_tx_pkt_burst(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Set IPv4 header in CPU-only packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param dst_addr Ethernet destination address
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_cpu_eth_hdr(AdvNetBurstParams *burst, int idx,
                                      char *dst_addr);
AdvNetStatus adv_net_set_cpu_eth_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                                      int idx,
                                      char *dst_addr);

/**
 * @brief Set IPv4 header in CPU-only packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param ip_len Length of packet after IPv4 header
 * @param proto L4 protocol
 * @param src_host Source host in host byte order
 * @param dst_host Destination host in host byte order
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_cpu_ipv4_hdr(AdvNetBurstParams *burst, int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host);
AdvNetStatus adv_net_set_cpu_ipv4_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                int idx,
                int ip_len,
                uint8_t proto,
                unsigned int src_host,
                unsigned int dst_host);

/**
 * @brief Set UDP header in CPU-only packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param udp_len Length of packet after UDP header
 * @param src_port Source port
 * @param dst_port Destination port
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_cpu_udp_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port);
AdvNetStatus adv_net_set_cpu_udp_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                int idx, int udp_len, uint16_t src_port, uint16_t dst_port);

/**
 * @brief Set UDP payload in CPU-only packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param data Payload data after UDP header
 * @param len Length of payload
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_cpu_udp_payload(AdvNetBurstParams *burst, int idx, void *data, int len);
AdvNetStatus adv_net_set_cpu_udp_payload(std::shared_ptr<AdvNetBurstParams> burst,
                int idx, void *data, int len);

/**
 * @brief Test if a TX burst is available
 *
 * Checks whether a TX burst for a given size can be allocated. This is useful for an
 * application to throttle its transmissions if the NIC and/or advanced network operator
 * is not keeping up with the desired rate. Rather than returning an error, the user can
 * use this function to loop or return later to try again.
 *
 * @param burst Info about burst of packets
 * @return true Burst is available
 * @return false Burst is not available
 */
bool adv_net_tx_burst_available(AdvNetBurstParams *burst);
bool adv_net_tx_burst_available(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Free all CPU packets and burst
 *
 * Frees every allocated CPU packets in the burst and the burst metadata. After this
 * call completes the CPU pointers are no longer valid.
 *
 * @param burst Burst to free
 */
void adv_net_free_cpu_pkts_and_burst(AdvNetBurstParams *burst);
void adv_net_free_cpu_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Free all packets and a burst
 *
 * Frees all packets in a burst of packets and the associated burst buffer
 *
 * @param burst Burst structure containing packet lists
 */
void adv_net_free_all_burst_pkts_and_burst(AdvNetBurstParams *burst);
void adv_net_free_all_burst_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Set packet length in metadata
 *
 * Sets metadata packet length. This is needed in addition to L3+L4 lengths for hardware
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 * @param cpu_len Length of CPU portion
 * @param gpu_len Length of GPU portion
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_pkt_len(AdvNetBurstParams *burst, int idx, int cpu_len, int gpu_len);
AdvNetStatus adv_net_set_pkt_len(std::shared_ptr<AdvNetBurstParams> burst,
                                    int idx,
                                    int cpu_len,
                                    int gpu_len);

/**
 * @brief Frees a single packet
 *
 * Frees a single packet from either the CPU or GPU buffer list. This function is extremely
 * inefficient since it frees a single packet, and bulk methods should be preferred instead.
 *
 * @param pkt Pointer to packet to free
 */
void adv_net_free_pkt(void *pkt);

/**
 * @brief Free all packets in a single list (CPU or GPU)
 *
 * Frees all packets in a single list. This function can be used when one type of packet
 * should be freed while the other is used for a longer period of time. For example, the
 * GPU buffers may be needed for pipeline processing, but the CPU buffers can be freed immediately
 * after sorting the packets in header-data spit mode.
 *
 * @param pkts List of packets
 * @param len Number of packets to free
 */
void adv_net_free_pkts(void **pkts, int len);

/**
 * @brief Free all packets in a burst
 *
 * Frees all packets in a burst of packets. After completion, all CPU and GPU packets will
 * be released back to the free pool.
 *
 * @param burst Burst structure containing packet lists
 */
void adv_net_free_all_burst_pkts(AdvNetBurstParams *burst);
void adv_net_free_all_burst_pkts(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Free a receive burst
 *
 * Frees the buffer containing a receive burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst
 */
void adv_net_free_rx_burst(AdvNetBurstParams *burst);
void adv_net_free_rx_burst(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Free a transmit burst buffer
 *
 * Frees the buffer containing a transmit burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst Burst structure to free
 */
void adv_net_free_tx_burst(AdvNetBurstParams *burst);
void adv_net_free_tx_burst(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Get the number of packets in a burst
 *
 * @param burst Burst structure with packets
 */
int64_t adv_net_get_num_pkts(AdvNetBurstParams *burst);
int64_t adv_net_get_num_pkts(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Get the queue ID of a burst
 *
 * @param burst Burst structure with packets
 */
int64_t adv_net_get_q_id(AdvNetBurstParams *burst);
int64_t adv_net_get_q_id(std::shared_ptr<AdvNetBurstParams> burst);

/**
 * @brief Set the number of packets in a burst
 *
 * @param burst Burst structure
 * @param num Number of packets
 */
void adv_net_set_num_pkts(AdvNetBurstParams *burst, int64_t num);
void adv_net_set_num_pkts(std::shared_ptr<AdvNetBurstParams> burst, int64_t num);

/**
 * @brief Set the header fields in a burst
 *
 * @param burst Burst structure
 * @param port Port ID of interface
 * @param q Queue ID of interface
 * @param num Number of packets
 */
void adv_net_set_hdr(AdvNetBurstParams *burst, uint16_t port, uint16_t q, int64_t num);
void adv_net_set_hdr(std::shared_ptr<AdvNetBurstParams> burst,
          uint16_t port, uint16_t q, int64_t num);

/**
 * @brief First MAC address string to char buffer
 *
 * @param dst Destination buffer
 * @param addr MAC address as string in format xx:xx:xx:xx:xx:xx
 */
void adv_net_format_eth_addr(char *dst, std::string addr);

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string &name);

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
};  // namespace holoscan::ops


template <>
struct YAML::convert<holoscan::ops::AdvNetConfigYaml> {
  static Node encode(const holoscan::ops::AdvNetConfigYaml& input_spec) {
    Node node;
    // node["type"] = inputTypeToString(input_spec.type_);
    // node["name"] = input_spec.tensor_name_;
    // node["opacity"] = std::to_string(input_spec.opacity_);
    // node["priority"] = std::to_string(input_spec.priority_);
    // node["color"] = input_spec.color_;
    // node["line_width"] = std::to_string(input_spec.line_width_);
    // node["point_size"] = std::to_string(input_spec.point_size_);
    // node["text"] = input_spec.text_;
    // node["depth_map_render_mode"] =
    //      depthMapRenderModeToString(input_spec.depth_map_render_mode_);
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::AdvNetConfigYaml& input_spec) {
    if (!node.IsMap()) {
      GXF_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      input_spec.common_.version        = node["version"].as<int32_t>();
      input_spec.common_.master_core_   = node["master_core"].as<int32_t>();

      try {
        const auto &rx = node["rx"];
        for (const auto &rx_item : rx) {
          holoscan::ops::AdvNetRxConfig rx_cfg;
          rx_cfg.if_name_ = rx_item["if_name"].as<std::string>();
          rx_cfg.flow_isolation_ = rx_item["flow_isolation"].as<bool>();

          for (const auto &q_item :  rx_item["queues"]) {
            holoscan::ops::RxQueueConfig q;
            q.common_.name_             = q_item["name"].as<std::string>();
            q.common_.id_               = q_item["id"].as<int>();
            q.common_.gpu_direct_       = q_item["gpu_direct"].as<bool>();
            if (q.common_.gpu_direct_) {
              q.common_.gpu_dev_          = q_item["gpu_device"].as<int>();
              q.common_.hds_              = q_item["split_boundary"].as<int>();
            }

            q.common_.cpu_cores_        = q_item["cpu_cores"].as<std::string>();
            q.common_.max_packet_size_  = q_item["max_packet_size"].as<int>();
            q.common_.num_concurrent_batches_  = q_item["num_concurrent_batches"].as<int>();
            q.common_.max_packet_size_  = q_item["max_packet_size"].as<int>();
            q.common_.batch_size_       = q_item["batch_size"].as<int>();
            q.output_port_              = q_item["output_port"].as<std::string>();

            rx_cfg.queues_.emplace_back(q);
          }

          for (const auto &flow_item :  rx_item["flows"]) {
            holoscan::ops::FlowConfig flow;
            flow.name_          = flow_item["name"].as<std::string>();

            flow.action_.type_     = holoscan::ops::FlowType::QUEUE;
            flow.action_.id_       = flow_item["action"]["id"].as<int>();
            flow.match_.udp_src_   = flow_item["match"]["udp_src"].as<uint16_t>();
            flow.match_.udp_dst_   = flow_item["match"]["udp_dst"].as<uint16_t>();

            rx_cfg.flows_.emplace_back(flow);
          }

          input_spec.rx_.emplace_back(rx_cfg);
        }

        const auto &tx = node["tx"];
        for (const auto &tx_item : tx) {
          holoscan::ops::AdvNetTxConfig tx_cfg;
          tx_cfg.if_name_ = tx_item["if_name"].as<std::string>();

          for (const auto &q_item :  tx_item["queues"]) {
            holoscan::ops::TxQueueConfig q;
            q.common_.name_             = q_item["name"].as<std::string>();
            q.common_.id_               = q_item["id"].as<int>();
            q.common_.gpu_direct_       = q_item["gpu_direct"].as<bool>();
            if (q.common_.gpu_direct_) {
              q.common_.gpu_dev_          = q_item["gpu_device"].as<int>();
              q.common_.hds_              = q_item["split_boundary"].as<int>();
            }

            q.common_.cpu_cores_        = q_item["cpu_cores"].as<std::string>();
            q.common_.max_packet_size_  = q_item["max_packet_size"].as<int>();
            q.common_.num_concurrent_batches_  = q_item["num_concurrent_batches"].as<int>();
            q.common_.max_packet_size_  = q_item["max_packet_size"].as<int>();
            q.common_.batch_size_       = q_item["batch_size"].as<int>();

            tx_cfg.queues_.emplace_back(q);
          }

          input_spec.tx_.emplace_back(tx_cfg);
        }
      } catch (const std::exception& e) {
        GXF_LOG_ERROR(e.what());
        return false;
      }

      HOLOSCAN_LOG_INFO("Finished reading advanced network operator config");

      return true;
    } catch (const std::exception& e) {
      GXF_LOG_ERROR(e.what());
      return false;
    }
  }
};
