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

#ifndef BURST_MANAGER_H_
#define BURST_MANAGER_H_

#include <cstddef>
#include <iostream>

#include "rmax_ano_data_types.h"
#include "rmax_service/ipo_chunk_consumer_base.h"
#include "rmax_service/rmax_ipo_receiver_service.h"
#include "adv_network_types.h"
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {
using namespace ral::services;

/**
 * @class RmaxBurst
 * @brief Represents a burst of packets in the advanced network.
 */
class RmaxBurst : public AdvNetBurstParams {
 public:
  static constexpr int MAX_PKT_IN_BURST = 9100;

  /**
   * @class BurstHandler
   * @brief Handles operations related to bursts.
   */
  class BurstHandler;

  /**
   * @brief Calculates the burst tag based on port ID and queue ID.
   *
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   * @return The calculated burst tag.
   */
  static inline uint32_t burst_tag_from_port_and_queue_id(int port_id, int queue_id) {
    return (port_id << 16) | queue_id;
  }

  /**
   * @brief Calculates the port ID based on the burst tag.
   *
   * @param tag The burst tag.
   * @return The calculated Port ID.
   */
  static inline uint16_t burst_port_id_from_burst_tag(uint32_t tag) {
    return (uint16_t)((tag >> 16) & 0xFFFF);
  }

  /**
   * @brief Calculates the queue ID based on the burst tag.
   *
   * @param tag The burst tag.
   * @return The calculated Queue ID.
   */
  static inline uint16_t burst_queue_id_from_burst_tag(uint32_t tag) {
    return (uint16_t)(tag & 0xFFFF);
  }

  /**
   * @brief Gets the burst ID.
   *
   * @return The burst ID.
   */
  inline uint16_t get_burst_id() const {
    auto burst_info = get_burst_info();
    if (burst_info == nullptr) { return 0; }
    return burst_info->burst_id;
  }

  /**
   * @brief Gets the port ID.
   *
   * @return The port ID.
   */
  inline uint16_t get_port_id() const { return hdr.hdr.port_id; }

  /**
   * @brief Gets the queue ID.
   *
   * @return The queue ID.
   */
  inline uint16_t get_queue_id() const { return hdr.hdr.q_id; }

  /**
   * @brief Gets the burst tag.
   *
   * @return The burst tag.
   */
  inline uint32_t get_burst_tag() const { return (hdr.hdr.port_id << 16) | hdr.hdr.q_id; }

  /**
   * @brief Checks if packet info is per packet.
   *
   * @return True if packet info is per packet, false otherwise.
   */
  inline bool is_packet_info_per_packet() const {
    return get_burst_flags() & BurstFlags::INFO_PER_PACKET;
  }

  /**
   * @brief Resets the burst packets
   */
  inline void reset_burst_packets() { set_num_packets(0); }

  /**
   * @brief Resets the burst and sets its parameters.
   *
   * @param hds_on Indicates if HDS is on.
   * @param header_stride_size The header stride size.
   * @param payload_stride_size The payload stride size.
   * @param gpu_direct Indicates if GPU direct is enabled.
   */
  inline void reset_burst_with_updated_params(bool hds_on, size_t header_stride_size,
                                              size_t payload_stride_size, bool gpu_direct) {
    set_num_packets(0);

    auto burst_info = get_burst_info();

    if (burst_info == nullptr) { return; }

    burst_info->hds_on = hds_on;
    burst_info->header_stride_size = header_stride_size;
    burst_info->payload_stride_size = payload_stride_size;
    burst_info->header_seg_idx = 0;
    burst_info->payload_on_cpu = !gpu_direct;

    if (hds_on) {
      burst_info->payload_seg_idx = 1;
      burst_info->header_on_cpu = true;
      hdr.hdr.num_segs = 2;
    } else {
      burst_info->payload_seg_idx = 0;
      burst_info->header_on_cpu = gpu_direct;
      hdr.hdr.num_segs = 1;
    }
  }

  /**
   * @brief Gets the flags of a burst.
   *
   * @return The flags of the burst.
   */
  inline BurstFlags get_burst_flags() const {
    auto burst_info = get_burst_info();
    if (burst_info == nullptr) { return FLAGS_NONE; }
    return burst_info->burst_flags;
  }

  /**
   * @brief Gets the extended info of a burst.
   *
   * @return A pointer to the extended info of the burst.
   */
  inline AnoBurstExtendedInfo* get_burst_info() const {
    return const_cast<AnoBurstExtendedInfo*>(
        reinterpret_cast<const AnoBurstExtendedInfo*>(&(hdr.custom_burst_data)));
  }

  /**
   * @brief Gets the maximum number of packets in a burst.
   *
   * @return The maximum number of packets in the burst.
   */
  inline uint16_t get_max_num_packets() const { return m_max_num_packets; }

  /**
   * @brief Gets the number of packets in a burst.
   *
   * @return The number of packets in the burst.
   */
  inline uint16_t get_num_packets() const { return hdr.hdr.num_pkts; }

  /**
   * @brief Appends a packet to the burst.
   *
   * @param packet_data The data of the packet to append.
   *
   * @throws runtime_error If the maximum number of packets is exceeded.
   */
  inline void append_packet(const RmaxPacketData& packet_data) {
    if (hdr.hdr.num_pkts >= m_max_num_packets) {
      throw std::runtime_error("Maximum number of packets exceeded (num_packets: " +
                               std::to_string(m_max_num_packets) + ")");
    }
    set_packet_data(get_num_packets(), packet_data);
    hdr.hdr.num_pkts += 1;
  }

 private:
  /**
   * @brief Appends a packet to the burst.
   *
   * @param packet_ind_in_out_burst The index of the packet in the burst.
   *                                Index boundary checks are not performed in this function.
   * @param packet_data The data of the packet to append.
   */
  inline void set_packet_data(size_t packet_ind_in_out_burst, const RmaxPacketData& packet_data) {
    auto burst_info = get_burst_info();

    if (burst_info->burst_flags & BurstFlags::INFO_PER_PACKET) {
      RmaxPacketExtendedInfo* rx_packet_info =
          reinterpret_cast<RmaxPacketExtendedInfo*>(pkt_extra_info[packet_ind_in_out_burst]);
      rx_packet_info->timestamp = packet_data.extended_info.timestamp;
      rx_packet_info->flow_tag = packet_data.extended_info.flow_tag;
    }

    if (burst_info->hds_on) {
      pkts[0][packet_ind_in_out_burst] = packet_data.header_ptr;
      pkts[1][packet_ind_in_out_burst] = packet_data.payload_ptr;
      pkt_lens[0][packet_ind_in_out_burst] = packet_data.header_length;
      pkt_lens[1][packet_ind_in_out_burst] = packet_data.payload_length;
    } else {
      pkts[0][packet_ind_in_out_burst] = packet_data.payload_ptr;
      pkts[1][packet_ind_in_out_burst] = nullptr;
      pkt_lens[0][packet_ind_in_out_burst] = packet_data.header_length;
      pkt_lens[1][packet_ind_in_out_burst] = 0;
    }
  }

  /**
   * @brief Sets the number of packets in a burst.
   *
   * @param num_pkts The number of packets to set.
   */
  inline void set_num_packets(uint16_t num_pkts) { hdr.hdr.num_pkts = num_pkts; }

  /**
   * @brief Constructs an RmaxBurst object.
   *
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   * @param max_packets_in_burst The maximum number of packets in the burst.
   */
  RmaxBurst(uint16_t port_id, uint16_t queue_id, uint16_t max_packets_in_burst = MAX_PKT_IN_BURST)
      : m_max_num_packets(max_packets_in_burst) {
    hdr.hdr.port_id = port_id;
    hdr.hdr.q_id = queue_id;
  }

 private:
  uint16_t m_max_num_packets = MAX_PKT_IN_BURST;
};

/**
 * @brief Class responsible for handling burst operations.
 */
class RmaxBurst::BurstHandler {
 public:
  /**
   * @brief Constructs a BurstHandler object.
   *
   * @param send_packet_ext_info Flag indicating whether to send packet info.
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   * @param gpu_direct Flag indicating whether GPU direct is enabled.
   */
  BurstHandler(bool send_packet_ext_info, int port_id, int queue_id, bool gpu_direct);

  /**
   * @brief Creates and initializes a new burst with the given burst ID
   *
   * @param burst_id The ID of the burst to create.
   * @return A shared pointer to the created burst.
   */
  std::shared_ptr<RmaxBurst> create_burst(uint16_t burst_id);

  /**
   * @brief Deletes a burst and frees its associated resources.
   *
   * @param burst A shared pointer to the burst to delete.
   */
  void delete_burst(std::shared_ptr<RmaxBurst> burst);

 private:
  bool m_send_packet_ext_info;
  int m_port_id;
  int m_queue_id;
  bool m_gpu_direct;
  AnoBurstExtendedInfo m_burst_info;
};

/**
 * @brief Manages RX bursts for advanced networking operations.
 *
 * The RxBurstsManager class is responsible for managing RX bursts in advanced networking
 * operations. It handles the creation, deletion, and processing of bursts, as well as
 * managing the lifecycle of packets within bursts. This class interfaces with the Rmax
 * framework to provide the necessary functionality for handling and transforming data
 * into a format suitable for ANO processing.
 */
class RxBurstsManager {
 public:
  static constexpr uint32_t DEFAULT_NUM_RX_BURSTS = 64;
  static constexpr uint32_t GET_BURST_TIMEOUT_MS = 1000;

  /**
   * @brief Constructor for the RxBurstsManager class.
   *
   * Initializes the burst manager with the specified parameters.
   *
   * @param send_packet_ext_info Flag indicating whether to send packet info.
   * @param port_id ID of the port.
   * @param queue_id ID of the queue.
   * @param burst_out_size Size of the burst output.
   * @param gpu_id ID of the GPU.
   * @param rx_bursts_out_queue Shared pointer to the output queue for RX bursts.
   */
  RxBurstsManager(bool send_packet_ext_info, int port_id, int queue_id, uint16_t burst_out_size = 0,
                  int gpu_id = INVALID_GPU_ID,
                  std::shared_ptr<IAnoBurstsCollection> rx_bursts_out_queue = nullptr);

  /**
   * @brief Destructor for the RxBurstsManager class.
   */
  virtual ~RxBurstsManager();

  /**
   * @brief Sets the parameters for the next chunk.
   *
   * @param chunk_size Size of the chunk.
   * @param hds_on Flag indicating if header data splitting (HDS) is enabled.
   * @param header_stride_size Stride size for the header data.
   * @param payload_stride_size Stride size for the payload data.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  inline ReturnStatus set_next_chunk_params(size_t chunk_size, bool hds_on,
                                            size_t header_stride_size, size_t payload_stride_size) {
    m_hds_on = hds_on;
    m_header_stride_size = header_stride_size;
    m_payload_stride_size = payload_stride_size;
    return ReturnStatus::success;
  }

  /**
   * @brief Submits the next packet to the burst manager.
   *
   * @param packet_data Extended information about the packet.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  inline ReturnStatus submit_next_packet(const RmaxPacketData& packet_data) {
    get_or_allocate_current_burst();
    if (m_cur_out_burst == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
      return ReturnStatus::no_free_chunks;
    }

    m_cur_out_burst->append_packet(packet_data);

    if (m_cur_out_burst->get_num_packets() >= m_burst_out_size) {
      return enqueue_and_reset_current_burst();
    }

    return ReturnStatus::success;
  }

  /**
   * @brief Gets an RX burst. Do not use in a case shared queue is used
   *
   * @param burst Pointer to the burst parameters.
   * @throws logic_error If shared output queue is used.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  inline ReturnStatus get_rx_burst(AdvNetBurstParams** burst) {
    if (m_using_shared_out_queue) {
      throw std::logic_error("Cannot get RX burst when using shared output queue");
    }

    auto out_burst = m_rx_bursts_out_queue->dequeue_burst().get();
    *burst = static_cast<AdvNetBurstParams*>(out_burst);
    if (*burst == nullptr) { return ReturnStatus::failure; }
    return ReturnStatus::success;
  }

  /**
   * @brief Marks the RX burst as done.
   *
   * @param burst Pointer to the burst parameters.
   */
  void rx_burst_done(RmaxBurst* burst);

 protected:
  /**
   * @brief Allocates a new burst.
   *
   * @return Shared pointer to the allocated burst parameters.
   */
  inline std::shared_ptr<RmaxBurst> allocate_burst() {
    auto burst = m_rx_bursts_mempool->dequeue_burst();
    return burst;
  }

  /**
   * @brief Gets or allocates the current burst.
   *
   * @return Shared pointer to the current burst parameters.
   */
  inline std::shared_ptr<RmaxBurst> get_or_allocate_current_burst() {
    if (m_cur_out_burst == nullptr) {
      m_cur_out_burst = allocate_burst();
      if (m_cur_out_burst == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
        return nullptr;
      }
      m_cur_out_burst->reset_burst_with_updated_params(
          m_hds_on, m_header_stride_size, m_payload_stride_size, m_gpu_direct);
    }
    return m_cur_out_burst;
  }

  inline ReturnStatus enqueue_and_reset_current_burst() {
    if (m_cur_out_burst == nullptr) {
      HOLOSCAN_LOG_ERROR("Trying to enqueue an empty burst");
      return ReturnStatus::failure;
    }

    bool res = m_rx_bursts_out_queue->enqueue_burst(m_cur_out_burst);
    reset_current_burst();
    if (!res) {
      HOLOSCAN_LOG_ERROR("Failed to enqueue burst");
      return ReturnStatus::failure;
    }

    return ReturnStatus::success;
  }

  /**
   * @brief Resets the current burst.
   */
  inline void reset_current_burst() { m_cur_out_burst = nullptr; }

 protected:
  bool m_send_packet_ext_info = false;
  int m_port_id = 0;
  int m_queue_id = 0;
  uint16_t m_burst_out_size = 0;
  int m_gpu_id = -1;
  bool m_hds_on = false;
  bool m_gpu_direct = false;
  size_t m_header_stride_size = 0;
  size_t m_payload_stride_size = 0;
  bool m_using_shared_out_queue = true;
  std::unique_ptr<IAnoBurstsCollection> m_rx_bursts_mempool = nullptr;
  std::shared_ptr<IAnoBurstsCollection> m_rx_bursts_out_queue = nullptr;
  std::shared_ptr<RmaxBurst> m_cur_out_burst = nullptr;
  AnoBurstExtendedInfo m_burst_info;
  std::unique_ptr<RmaxBurst::BurstHandler> m_burst_handler;
};

};  // namespace holoscan::ops

#endif /* BURST_MANAGER_H_ */
