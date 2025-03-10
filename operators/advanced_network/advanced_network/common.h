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
#include <unordered_set>
#include <memory>
#include <optional>
#include <tuple>
#include <stdint.h>
#include <yaml-cpp/yaml.h>
#include "advanced_network/types.h"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

// this part is purely optional, just a helper for the user
AdvNetBurstParams* adv_net_create_burst_params();

namespace detail {
inline AdvNetDirection DirectionStringToType(const std::string& dir) {
  if (dir == "rx") {
    return AdvNetDirection::RX;
  } else if (dir == "tx") {
    return AdvNetDirection::TX;
  }

  return AdvNetDirection::TX_RX;
}
};  // namespace detail

/**
 * @brief Determine which directions are enabled
 *
 * @param dir Direction from config. Either "rx", "tx", or "tx/rx"
 * @return int Number of directions enabled
 */
inline int EnabledDirections(const std::string& dir) {
  if (dir == "rx" || dir == "tx") { return 1; }

  return 0;
}

/**
 * @brief Returns a manager type
 *
 * @return Manager type
 */
AnoMgrType adv_net_get_manager_type();

/**
 * @brief Returns a manager type
 *
 * @param config YML Configuration structure (e.g. AdvNetConfigYaml)
 * @return Manager type
 */
template <typename Config>
AnoMgrType adv_net_get_manager_type(const Config& config);

/**
 * @brief Returns a raw packet pointer from a pointer in AdvNetBurstParams
 *
 * The AdvNetBurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the CPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param seg Segment of packet
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void* adv_net_get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx);

/**
 * @brief Returns a raw packet pointer from a pointer in AdvNetBurstParams
 *
 * The AdvNetBurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the GPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void* adv_net_get_pkt_ptr(AdvNetBurstParams* burst, int idx);

/**
 * @brief Get packet length of a segment of a packet
 *
 * @param burst Burst structure containing packets
 * @param seg Segment of packet
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint16_t adv_net_get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx);

/**
 * @brief Get packet length of an entire packet
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint16_t adv_net_get_pkt_len(AdvNetBurstParams* burst, int idx);

/**
 * @brief Get flow ID of a packet
 *
 * Retrieves the flow ID of a packet, or 0 if no flow was matched. The flow ID should match
 * the flow ID in the flow rule for the ANO config.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Flow ID
 */
uint16_t adv_net_get_pkt_flow_id(AdvNetBurstParams* burst, int idx);

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
AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams* burst);

/**
 * @brief Set IPv4 header in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param dst_addr Ethernet destination address
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr);

/**
 * @brief Set IPv4 header in packet
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
AdvNetStatus adv_net_set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                  unsigned int src_host, unsigned int dst_host);

/**
 * @brief Set UDP header in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param udp_len Length of packet after UDP header
 * @param src_port Source port
 * @param dst_port Destination port
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                 uint16_t dst_port);

/**
 * @brief Set UDP payload in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param data Payload data after UDP header
 * @param len Length of payload
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len);

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
bool adv_net_tx_burst_available(AdvNetBurstParams* burst);

/**
 * @brief Free all packets and burst from one segment
 *
 * Frees every allocated packets in the burst and the burst metadata for one segment.
 * After this call completes the segment's pointers are no longer valid.
 *
 * @param burst Burst to free
 */
void adv_net_free_seg_pkts_and_burst(AdvNetBurstParams* burst, int seg);

/**
 * @brief Free all packets and a burst
 *
 * Frees all packets in a burst of packets and the associated burst buffer
 *
 * @param burst Burst structure containing packet lists
 */
void adv_net_free_all_pkts_and_burst(AdvNetBurstParams* burst);

/**
 * @brief Set packet lengths in metadata
 *
 * Sets metadata packet lengths. This is needed in addition to L3+L4 lengths for hardware
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 * @param lens Lengths of each segment
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
AdvNetStatus adv_net_set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                  const std::initializer_list<int>& lens);

/**
 * @brief Set packet TX time
 *
 * Sets the transmit time (in PTP time) to transmit the packet. Every packet transmitted
 * after this one in the same queue will be transmitted no earlier than the time listed
 * in the function call. This feature is only available on ConnectX-7 or BlueField 3 and
 * higher cards.
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 * @param time PTP time to transmit
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Time set successfully
 */
AdvNetStatus adv_net_set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t time);


uint64_t adv_net_get_burst_tot_byte(AdvNetBurstParams* burst);

/**
 * @brief Frees all segments of a single packet
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 */
void adv_net_free_pkt(AdvNetBurstParams* burst, int idx);


/**
 * @brief Frees a single segment from a single packet
 *
 * @param burst Burst structure containing packet lists
 * @param seg Segment of packet in scatter list
 * @param idx Index of packet
 */
void adv_net_free_pkt_seg(AdvNetBurstParams* burst, int seg, int idx);

/**
 * @brief Free all packets for a single segment in a burst
 *
 * Frees all packets in a single segment in a burst of packets.
 *
 * @param burst Burst structure containing packet lists
 */
void adv_net_free_all_seg_pkts(AdvNetBurstParams* burst, int seg);

/**
 * @brief Free a receive burst
 *
 * Frees the buffer containing a receive burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst
 */
void adv_net_free_rx_burst(AdvNetBurstParams* burst);

/**
 * @brief Free a transmit burst buffer
 *
 * Frees the buffer containing a transmit burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst Burst structure to free
 */
void adv_net_free_tx_burst(AdvNetBurstParams* burst);

/**
 * @brief Get the number of packets in a burst
 *
 * @param burst Burst structure with packets
 */
int64_t adv_net_get_num_pkts(AdvNetBurstParams* burst);

/**
 * @brief Get the queue ID of a burst
 *
 * @param burst Burst structure with packets
 */
int64_t adv_net_get_q_id(AdvNetBurstParams* burst);

/**
 * @brief Get mac address of an interface
 *
 * @param addr Address of interface from config file
 * @param port Port ID of interface
 *
 * @returns AdvNetStatus::SUCCESS on success
 */
AdvNetStatus adv_net_get_mac(int port, char* mac);

/**
 * @brief Get mac address of an interface
 *
 * @param addr Address of interface from config file
 *
 * @returns Port number or -1 for not found
 */
int adv_net_address_to_port(const std::string& addr);

/**
 * @brief Set the number of packets in a burst
 *
 * @param burst Burst structure
 * @param num Number of packets
 */
void adv_net_set_num_pkts(AdvNetBurstParams* burst, int64_t num);

/**
 * @brief Set the header fields in a burst
 *
 * @param burst Burst structure
 * @param port Port ID of interface
 * @param q Queue ID of interface
 * @param num Number of packets
 * @param segs Number of segments
 */
void adv_net_set_hdr(AdvNetBurstParams* burst, uint16_t port, uint16_t q, int64_t num, int segs);

/**
 * @brief First MAC address string to char buffer
 *
 * @param dst Destination buffer
 * @param addr MAC address as string in format xx:xx:xx:xx:xx:xx
 */
void adv_net_format_eth_addr(char* dst, std::string addr);

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string& name);

/**
 * @brief Shut down ANO and do any cleanup necessary. Freeing memory is done
 * in the manager's destructor.
 *
 */
void adv_net_shutdown();

/**
 * @brief Print port statistics
 *
 */
void adv_net_print_stats();

/**
 * @brief Get the list (set) of rx/tx ports from a node
 *
 * @param node Yaml node
 * @param dir String of direction ["rx", "tx"]
 *
 * @returns unordered set of rx/tx port names
 */
std::unordered_set<std::string> adv_net_get_port_names(const Config& conf, const std::string& dir);

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
  /**
   * @brief Parse flow configuration from a YAML node.
   *
   * @param flow_item The YAML node containing the flow configuration.
   * @param flow The FlowConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_flow_config(const YAML::Node& flow_item, holoscan::ops::FlowConfig& flow);

  /**
   * @brief Parse memory region configuration from a YAML node.
   *
   * @param mr The YAML node containing the memory region configuration.
   * @param tmr The MemoryRegion object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_memory_region_config(const YAML::Node& mr,
                                         holoscan::ops::MemoryRegion& memory_region);

  /**
   * @brief Parse common RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_config(const YAML::Node& q_item,
                                    const holoscan::ops::AnoMgrType& manager_type,
                                    holoscan::ops::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param manager_type The manager type.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_common_config(const YAML::Node& q_item,
                                           holoscan::ops::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse common TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_config(const YAML::Node& q_item,
                                    const holoscan::ops::AnoMgrType& manager_type,
                                    holoscan::ops::TxQueueConfig& tx_queue_config);

  /**
   * @brief Parse TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param manager_type The manager type.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_common_config(const YAML::Node& q_item,
                                           holoscan::ops::TxQueueConfig& tx_queue_config);

  /**
   * @brief Decode the YAML node into an AdvNetConfigYaml object.
   *
   * This function parses the provided YAML node and populates the given AdvNetConfigYaml object.
   * It handles various configurations such as version, master core, manager type, debug flag,
   * memory regions, interfaces, RX queues, TX queues, and flows.
   *
   * @param node The YAML node containing the configuration.
   * @param input_spec The AdvNetConfigYaml object to populate.
   * @return true if decoding was successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::ops::AdvNetConfigYaml& input_spec) {
    if (!node.IsMap()) {
      GXF_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      input_spec.common_.version = node["version"].as<int32_t>();
      input_spec.common_.master_core_ = node["master_core"].as<int32_t>();
      try {
        input_spec.common_.manager_type = holoscan::ops::manager_type_from_string(
            node["manager"].as<std::string>(holoscan::ops::ANO_MGR_STR__DEFAULT));
      } catch (const std::exception& e) {
        input_spec.common_.manager_type = holoscan::ops::AnoMgrType::DEFAULT;
      }

      try {
        input_spec.debug_ = node["debug"].as<bool>(false);
      } catch (const std::exception& e) { input_spec.debug_ = false; }

      try {
        input_spec.log_level_ =
            holoscan::ops::AnoLogLevel::from_string(node["log_level"].as<std::string>(
                holoscan::ops::AnoLogLevel::to_string(holoscan::ops::AnoLogLevel::WARN)));
      } catch (const std::exception& e) {
        input_spec.log_level_ = holoscan::ops::AnoLogLevel::WARN;
      }

      try {
        const auto& mrs = node["memory_regions"];
        for (const auto& mr : mrs) {
          holoscan::ops::MemoryRegion tmr;
          if (!parse_memory_region_config(mr, tmr)) {
            HOLOSCAN_LOG_ERROR("Failed to parse memory region config");
            return false;
          }
          if (input_spec.mrs_.find(tmr.name_) != input_spec.mrs_.end()) {
            HOLOSCAN_LOG_CRITICAL("Duplicate memory region names: {}", tmr.name_);
            return false;
          }
          input_spec.mrs_[tmr.name_] = tmr;
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Must define at least one memory type");
        return false;
      }

      try {
        const auto& intfs = node["interfaces"];
        for (const auto& intf : intfs) {
          holoscan::ops::AdvNetConfigInterface ifcfg;

          ifcfg.name_ = intf["name"].as<std::string>();
          ifcfg.address_ = intf["address"].as<std::string>();
          try {
            ifcfg.flow_isolation_ = intf["flow_isolation"].as<bool>();
          } catch (const std::exception& e) { ifcfg.flow_isolation_ = false; }

          try {
            const auto& rx = intf["rx"];
            holoscan::ops::AdvNetRxConfig rx_cfg;

            for (const auto& q_item : rx["queues"]) {
              holoscan::ops::RxQueueConfig q;
              if (!parse_rx_queue_config(q_item, input_spec.common_.manager_type, q)) {
                HOLOSCAN_LOG_ERROR("Failed to parse RxQueueConfig");
                return false;
              }

              try {
                q.timeout_us_ = q_item["timeout_us"].as<uint64_t>();
              } catch (const std::exception& e) { q.timeout_us_ = 0; }

              rx_cfg.queues_.emplace_back(std::move(q));
            }

            for (const auto& flow_item : rx["flows"]) {
              holoscan::ops::FlowConfig flow;
              if (!parse_flow_config(flow_item, flow)) {
                HOLOSCAN_LOG_ERROR("Failed to parse FlowConfig");
                return false;
              }
              rx_cfg.flows_.emplace_back(std::move(flow));
            }

            ifcfg.rx_ = rx_cfg;
          } catch (const std::exception& e) {}  // No RX queues defined for this interface.

          try {
            const auto& tx = intf["tx"];
            holoscan::ops::AdvNetTxConfig tx_cfg;

            try {
              tx_cfg.accurate_send_ = tx["accurate_send"].as<bool>();
            } catch (const std::exception& e) { tx_cfg.accurate_send_ = false; }

            for (const auto& q_item : tx["queues"]) {
              holoscan::ops::TxQueueConfig q;
              if (!parse_tx_queue_config(q_item, input_spec.common_.manager_type, q)) {
                HOLOSCAN_LOG_ERROR("Failed to parse TxQueueConfig");
                return false;
              }
              tx_cfg.queues_.emplace_back(std::move(q));
            }

            ifcfg.tx_ = tx_cfg;
          } catch (const std::exception& e) {}  // No TX queues defined for this interface.

          input_spec.ifs_.push_back(ifcfg);
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
