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

namespace holoscan::advanced_network {

// this part is purely optional, just a helper for the user
BurstParams* create_burst_params();
BurstParams* create_tx_burst_params();

enum class ErrorGlobalStats {
  OUT_OF_RX_BUFFERS = 0,
  RX_QUEUE_FULL = 1,
  METADATA_BUF_DEPLETED = 2,

  SENTINEL = 3,
};

namespace detail {
inline Direction DirectionStringToType(const std::string& dir) {
  if (dir == "rx") {
    return Direction::RX;
  } else if (dir == "tx") {
    return Direction::TX;
  }

  return Direction::TX_RX;
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
 * @brief Initialize the backend manager and any other resources needed
 *
 * @param config YML Configuration structure (e.g. AdvNetConfigYaml)
 * @return AdvNetStatus indicating status. Valid values are:
 *    SUCCESS: Initialization successful
 *    INVALID_CONFIG: Invalid configuration
 *    INTERNAL_ERROR: Internal error
 */
Status adv_net_init(NetworkConfig& config);

/**
 * @brief Returns a manager type
 *
 * @return Manager type
 */
ManagerType get_manager_type();

/**
 * @brief Returns a manager type
 *
 * @param config YML Configuration structure (e.g. NetworkConfig)
 * @return Manager type
 */
template <typename Config>
ManagerType get_manager_type(const Config& config);

/**
 * @brief Returns a raw packet pointer from a pointer in BurstParams
 *
 * The BurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the CPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param seg Segment of packet
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx);

/**
 * @brief Returns a raw packet pointer from a pointer in BurstParams
 *
 * The BurstParams structure contains pointers to opaque packets which are not accessible
 * directly by the user. This function fetches the GPU packet pointer at index idx
 * from the burst.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return Pointer to packet data
 */
void* get_packet_ptr(BurstParams* burst, int idx);

/**
 * @brief Get packet length of a segment of a packet
 *
 * @param burst Burst structure containing packets
 * @param seg Segment of packet
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx);

/**
 * @brief Get packet length of an entire packet
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Length of packet
 */
uint32_t get_packet_length(BurstParams* burst, int idx);

/**
 * @brief Get flow ID of a packet
 *
 * Retrieves the flow ID of a packet, or 0 if no flow was matched. The flow ID should match
 * the flow ID in the flow rule for the advanced_network config.
 *
 * @param burst Burst structure containing packets
 * @param idx Index of packet
 * @return uint16_t Flow ID
 */
uint16_t get_packet_flow_id(BurstParams* burst, int idx);

/**
 * @brief Populate a TX packet burst buffer
 *
 * Populates a transmit packet burst buffer with allocated packets. The user can take these
 * allocated packets and fill with the desired data/headers.
 *
 * @param burst Burst structure to populate
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packets allocated
 *    NULL_PTR: Burst or packet pools uninitialized
 *    NO_FREE_BURST_BUFFERS: No burst buffers to allocate
 *    NO_FREE_CPU_PACKET_BUFFERS: Not enough CPU packet buffers available
 */
Status get_tx_packet_burst(BurstParams* burst);

/**
 * @brief Set IPv4 header in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param dst_addr Ethernet destination address
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
Status set_eth_header(BurstParams* burst, int idx, char* dst_addr);

/**
 * @brief Set IPv4 header in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param ip_len Length of packet after IPv4 header
 * @param proto L4 protocol
 * @param src_host Source host in host byte order
 * @param dst_host Destination host in host byte order
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                       unsigned int src_host, unsigned int dst_host);

/**
 * @brief Set UDP header in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param udp_len Length of packet after UDP header
 * @param src_port Source port
 * @param dst_port Destination port
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                      uint16_t dst_port);

/**
 * @brief Set UDP payload in packet
 *
 * @param burst Burst structure to populate
 * @param idx Index of packet
 * @param data Payload data after UDP header
 * @param len Length of payload
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
Status set_udp_payload(BurstParams* burst, int idx, void* data, int len);

/**
 * @brief Test if a TX burst is available
 *
 * Checks whether a TX burst for a given size can be allocated. This is useful for an
 * application to throttle its transmissions if the NIC is not keeping up with the desired rate.
 * Rather than returning an error, the user can use this function to loop or return later
 * to try again.
 *
 * @param burst Info about burst of packets
 * @return true Burst is available
 * @return false Burst is not available
 */
bool is_tx_burst_available(BurstParams* burst);

/**
 * @brief Free all packets and burst from one segment
 *
 * Frees every allocated packets in the burst and the burst metadata for one segment.
 * After this call completes the segment's pointers are no longer valid.
 *
 * @param burst Burst to free
 */
void free_segment_packets_and_burst(BurstParams* burst, int seg);

/**
 * @brief Free all packets and an RX burst
 *
 * Frees all packets in a burst of packets and the associated burst buffer
 *
 * @param burst Burst structure containing packet lists
 */
void free_all_packets_and_burst_rx(BurstParams* burst);

/**
 * @brief Free all packets and a TX burst
 *
 * Frees all packets in a burst of packets and the associated burst buffer
 *
 * @param burst Burst structure containing packet lists
 */
void free_all_packets_and_burst_tx(BurstParams* burst);

/**
 * @brief Set packet lengths in metadata
 *
 * Sets metadata packet lengths. This is needed in addition to L3+L4 lengths for hardware
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 * @param lens Lengths of each segment
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Packet populated successfully
 */
Status set_packet_lengths(BurstParams* burst, int idx, const std::initializer_list<int>& lens);

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
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Time set successfully
 */
Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t time);

uint64_t get_burst_tot_byte(BurstParams* burst);

/**
 * @brief Frees all segments of a single packet
 *
 * @param burst Burst structure containing packet lists
 * @param idx Index of packet
 */
void free_packet(BurstParams* burst, int idx);

/**
 * @brief Frees a single segment from a single packet
 *
 * @param burst Burst structure containing packet lists
 * @param seg Segment of packet in scatter list
 * @param idx Index of packet
 */
void free_packet_segment(BurstParams* burst, int seg, int idx);

/**
 * @brief Free all packets for a single segment in a burst
 *
 * Frees all packets in a single segment in a burst of packets.
 *
 * @param burst Burst structure containing packet lists
 */
void free_all_segment_packets(BurstParams* burst, int seg);

/**
 * @brief Free a receive burst
 *
 * Frees the buffer containing a receive burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst
 */
void free_rx_burst(BurstParams* burst);

/**
 * @brief Free a transmit burst buffer
 *
 * Frees the buffer containing a transmit burst buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst Burst structure to free
 */
void free_tx_burst(BurstParams* burst);

/**
 * @brief Free a receive TX meta buffer
 *
 * Frees the buffer containing a receive TX meta buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst Burst structure to free
 */
void free_tx_metadata(BurstParams* burst);

/**
 * @brief Free a receive RX meta buffer
 *
 * Frees the buffer containing a receive RX meta buffer. This function does not free packets;
 * packets must be freed prior to calling this.
 *
 * @param burst Burst structure to free
 */
void free_rx_metadata(BurstParams* burst);

/**
 * @brief Get the number of packets in a burst
 *
 * @param burst Burst structure with packets
 */
int64_t get_num_packets(BurstParams* burst);

/**
 * @brief Get the queue ID of a burst
 *
 * @param burst Burst structure with packets
 */
int64_t get_q_id(BurstParams* burst);

/**
 * @brief Get mac address of an interface
 *
 * @param port Port number of interface
 * @param mac MAC address of interface
 *
 * @returns Status::SUCCESS on success
 */
Status get_mac_addr(int port, char* mac);

/**
 * @brief Get port number from interface name
 *
 * @param key PCIe address or config name of the interface to look up
 *
 * @returns Port number or -1 for not found
 */
int get_port_id(const std::string& key);

/**
 * @brief Set the number of packets in a burst
 *
 * @param burst Burst structure
 * @param num Number of packets
 */
void set_num_packets(BurstParams* burst, int64_t num);

/**
 * @brief Send a TX burst
 *
 * @param burst Burst structure
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Burst sent successfully
 */
Status send_tx_burst(BurstParams* burst);

/**
 * @brief Get a RX burst
 *
 * @param burst Burst structure
 * @param port Port ID of interface
 * @param q Queue ID of interface
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Burst received successfully
 *    NULL_PTR: No bursts ready to receive
 */
Status get_rx_burst(BurstParams** burst, int port, int q);

/**
 * @brief Get a RX burst from any queue on a specific port
 *
 * @param burst Burst structure
 * @param port Port ID of interface
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Burst received successfully
 *    NULL_PTR: No bursts ready to receive on any queue for this port
 */
Status get_rx_burst(BurstParams** burst, int port);

/**
 * @brief Get a RX burst from any queue on any port
 *
 * @param burst Burst structure
 * @return Status indicating status. Valid values are:
 *    SUCCESS: Burst received successfully
 *    NULL_PTR: No bursts ready to receive on any queue on any port
 */
Status get_rx_burst(BurstParams** burst);

/**
 * @brief Get a RX burst
 *
 * @param burst Burst structure
 * @param conn_id Connection ID
 * @param server True if server, false if client
 */
Status get_rx_burst(BurstParams** burst, uintptr_t conn_id, bool server);

/**
 * @brief Set the header fields in a burst
 *
 * @param burst Burst structure
 * @param port Port ID of interface
 * @param q Queue ID of interface
 * @param num Number of packets
 * @param segs Number of segments
 */
void set_header(BurstParams* burst, uint16_t port, uint16_t q, int64_t num, int segs);

/**
 * @brief First MAC address string to char buffer
 *
 * @param dst Destination buffer
 * @param addr MAC address as string in format xx:xx:xx:xx:xx:xx
 */
void format_eth_addr(char* dst, std::string addr);

/**
 * @brief Shut down the advanced_network and do any cleanup necessary. Freeing memory is done
 * in the manager's destructor.
 *
 */
void shutdown();

/**
 * @brief Print port statistics
 *
 */
void print_stats();

/**
 * @brief Get the number of RX queues. May be overridden by the manager if the number of queues
 * differs from what is defined in the config.
 *
 * @param port_id Port ID of interface
 * @return uint16_t Number of RX queues
 */
uint16_t get_num_rx_queues(int port_id);

// RDMA functions
Status rdma_connect_to_server(const std::string& server_addr, uint16_t server_port,
                              uintptr_t* conn_id);
Status rdma_connect_to_server(const std::string& server_addr, uint16_t server_port,
                              const std::string& src_addr, uintptr_t* conn_id);
Status rdma_get_port_queue(uintptr_t conn_id, uint16_t* port, uint16_t* queue);
Status rdma_get_server_conn_id(const std::string& server_addr, uint16_t server_port,
                               uintptr_t* conn_id);
Status rdma_set_header(BurstParams* burst, RDMAOpCode op_code, uintptr_t conn_id, bool is_server,
                       int num_pkts, uint64_t wr_id, const std::string& local_mr_name);
RDMAOpCode rdma_get_opcode(BurstParams* burst);

};  // namespace holoscan::advanced_network

template <>
struct YAML::convert<holoscan::advanced_network::NetworkConfig> {
  static Node encode(const holoscan::advanced_network::NetworkConfig& input_spec) {
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
  static bool parse_flow_config(const YAML::Node& flow_item,
                                holoscan::advanced_network::FlowConfig& flow);

  /**
   * @brief Parse memory region configuration from a YAML node.
   *
   * @param mr The YAML node containing the memory region configuration.
   * @param tmr The MemoryRegionConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_memory_region_config(
      const YAML::Node& mr, holoscan::advanced_network::MemoryRegionConfig& memory_region);

  /**
   * @brief Parse common RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_config(const YAML::Node& q_item,
                                    const holoscan::advanced_network::ManagerType& manager_type,
                                    holoscan::advanced_network::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param manager_type The manager type.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_common_config(
      const YAML::Node& q_item, holoscan::advanced_network::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse common TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_config(const YAML::Node& q_item,
                                    const holoscan::advanced_network::ManagerType& manager_type,
                                    holoscan::advanced_network::TxQueueConfig& tx_queue_config);

  /**
   * @brief Parse TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param manager_type The manager type.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_common_config(
      const YAML::Node& q_item, holoscan::advanced_network::TxQueueConfig& tx_queue_config);

  /**
   * @brief Decode the YAML node into an NetworkConfig object.
   *
   * This function parses the provided YAML node and populates the given NetworkConfig object.
   * It handles various configurations such as version, master core, manager type, debug flag,
   * memory regions, interfaces, RX queues, TX queues, and flows.
   *
   * @param node The YAML node containing the configuration.
   * @param input_spec The NetworkConfig object to populate.
   * @return true if decoding was successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::advanced_network::NetworkConfig& input_spec) {
    if (!node.IsMap()) {
      GXF_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      input_spec.common_.version = node["version"].as<int32_t>();
      input_spec.common_.master_core_ = node["master_core"].as<int32_t>();
      try {
        input_spec.common_.manager_type = holoscan::advanced_network::manager_type_from_string(
            node["manager"].as<std::string>(holoscan::advanced_network::ANO_MGR_STR__DEFAULT));
      } catch (const std::exception& e) {
        input_spec.common_.manager_type = holoscan::advanced_network::ManagerType::DEFAULT;
      }

      try {
        input_spec.debug_ = node["debug"].as<bool>(false);
      } catch (const std::exception& e) { input_spec.debug_ = false; }

      try {
        input_spec.log_level_ = holoscan::advanced_network::LogLevel::from_string(
            node["log_level"].as<std::string>(holoscan::advanced_network::LogLevel::to_string(
                holoscan::advanced_network::LogLevel::WARN)));
      } catch (const std::exception& e) {
        input_spec.log_level_ = holoscan::advanced_network::LogLevel::WARN;
      }

      try {
        const auto& mrs = node["memory_regions"];
        for (const auto& mr : mrs) {
          holoscan::advanced_network::MemoryRegionConfig tmr;
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
          holoscan::advanced_network::InterfaceConfig ifcfg;

          ifcfg.name_ = intf["name"].as<std::string>();
          ifcfg.address_ = intf["address"].as<std::string>();

          // RDMA config
          try {
            ifcfg.rdma_.mode_ = holoscan::advanced_network::GetRDMAModeFromString(
                intf["rdma_mode"].as<std::string>());
            ifcfg.rdma_.xmode_ = holoscan::advanced_network::GetRDMATransportModeFromString(
                intf["rdma_transport_mode"].as<std::string>());
            ifcfg.rdma_.port_ = intf["rdma_port"].as<uint16_t>();
          } catch (const std::exception& e) {
            // Non-RDMA config
          }

          try {
            const auto& rx = intf["rx"];
            holoscan::advanced_network::RxConfig rx_cfg;

            try {
              rx_cfg.flow_isolation_ = rx["flow_isolation"].as<bool>();
            } catch (const std::exception& e) { rx_cfg.flow_isolation_ = false; }

            for (const auto& q_item : rx["queues"]) {
              holoscan::advanced_network::RxQueueConfig q;
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
              holoscan::advanced_network::FlowConfig flow;
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
            holoscan::advanced_network::TxConfig tx_cfg;

            try {
              tx_cfg.accurate_send_ = tx["accurate_send"].as<bool>();
            } catch (const std::exception& e) { tx_cfg.accurate_send_ = false; }

            for (const auto& q_item : tx["queues"]) {
              holoscan::advanced_network::TxQueueConfig q;
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

      HOLOSCAN_LOG_INFO("Finished reading Advanced Network configuration");

      return true;
    } catch (const std::exception& e) {
      GXF_LOG_ERROR(e.what());
      return false;
    }
  }
};
