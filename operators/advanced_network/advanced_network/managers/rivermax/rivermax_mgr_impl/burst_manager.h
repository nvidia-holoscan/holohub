/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/logger/logger.hpp>

#include "rdk/services/services.h"

#include "advanced_network/types.h"
#include "rivermax_ano_data_types.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::services;

/**
 * @class RivermaxBurst
 * @brief Represents a burst of packets in the advanced network.
 */
class RivermaxBurst : public BurstParams {
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
    return static_cast<BurstFlags>(hdr.hdr.burst_flags);
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
  inline uint16_t get_max_num_packets() const { return max_num_packets_; }

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
  inline void append_packet(const RivermaxPacketData& packet_data) {
    if (hdr.hdr.num_pkts >= max_num_packets_) {
      throw std::runtime_error("Maximum number of packets exceeded (num_packets: " +
                               std::to_string(max_num_packets_) + ")");
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
  inline void set_packet_data(size_t packet_ind_in_out_burst,
                              const RivermaxPacketData& packet_data) {
    auto burst_info = get_burst_info();

    if (get_burst_flags() & BurstFlags::INFO_PER_PACKET) {
      RivermaxPacketExtendedInfo* rx_packet_info =
          reinterpret_cast<RivermaxPacketExtendedInfo*>(pkt_extra_info[packet_ind_in_out_burst]);
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
   * @brief Constructs an RivermaxBurst object.
   *
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   * @param max_packets_in_burst The maximum number of packets in the burst.
   */
  RivermaxBurst(uint16_t port_id, uint16_t queue_id,
                uint16_t max_packets_in_burst = MAX_PKT_IN_BURST)
      : max_num_packets_(max_packets_in_burst) {
    hdr.hdr.port_id = port_id;
    hdr.hdr.q_id = queue_id;
  }

 private:
  uint16_t max_num_packets_ = MAX_PKT_IN_BURST;
};

/**
 * @brief Class responsible for handling burst operations.
 */
class RivermaxBurst::BurstHandler {
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
  std::shared_ptr<RivermaxBurst> create_burst(uint16_t burst_id);

  /**
   * @brief Deletes a burst and frees its associated resources.
   *
   * @param burst A shared pointer to the burst to delete.
   */
  void delete_burst(std::shared_ptr<RivermaxBurst> burst);

 private:
  bool send_packet_ext_info_;
  int port_id_;
  int queue_id_;
  bool gpu_direct_;
  AnoBurstExtendedInfo burst_info_;
};

/**
 * @brief Manages RX bursts for advanced networking operations.
 *
 * The RxBurstsManager class is responsible for managing RX bursts in advanced networking
 * operations. It handles the creation, deletion, and processing of bursts, as well as
 * managing the lifecycle of packets within bursts. This class interfaces with the Rivermax
 * framework to provide the necessary functionality for handling and transforming data
 * into a format suitable for advanced_network to process.
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
   *                            If not provided a local queue will be used.
   */
  RxBurstsManager(bool send_packet_ext_info, int port_id, int queue_id, uint16_t burst_out_size = 0,
                  int gpu_id = INVALID_GPU_ID,
                  std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>> rx_bursts_out_queue = {});

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
   * @return Status indicating the success or failure of the operation.
   */
  inline Status set_next_chunk_params(size_t chunk_size, bool hds_on, size_t header_stride_size,
                                      size_t payload_stride_size) {
    hds_on_ = hds_on;
    header_stride_size_ = header_stride_size;
    payload_stride_size_ = payload_stride_size;
    return Status::SUCCESS;
  }

  /**
   * @brief Submits the next packet to the burst manager.
   *
   * @param packet_data Extended information about the packet.
   * @return Status indicating the success or failure of the operation.
   */
  inline Status submit_next_packet(int stream_id, const RivermaxPacketData& packet_data) {
    get_or_allocate_current_burst(stream_id);
    if (cur_out_burst_[stream_id] == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
      return Status::NO_FREE_BURST_BUFFERS;
    }

    cur_out_burst_[stream_id]->append_packet(packet_data);

    if (cur_out_burst_[stream_id]->get_num_packets() >= burst_out_size_) {
      return enqueue_and_reset_current_burst(stream_id);
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Gets an RX burst. Do not use in a case shared queue is used
   *
   * @param burst Pointer to the burst parameters.
   * @throws logic_error If shared output queue is used.
   * @return Status indicating the success or failure of the operation.
   */
  inline Status get_rx_burst(BurstParams** burst, int stream_id) {
    if (using_shared_out_queue_) {
      throw std::logic_error("Cannot get RX burst when using shared output queue");
    }

    auto out_burst = rx_bursts_out_queue_[stream_id]->dequeue_burst().get();
    *burst = static_cast<BurstParams*>(out_burst);
    if (*burst == nullptr) { return Status::NULL_PTR; }
    return Status::SUCCESS;
  }

  /**
   * @brief Marks the RX burst as done.
   *
   * @param burst Pointer to the burst parameters.
   */
  void rx_burst_done(RivermaxBurst* burst);

 protected:
  /**
   * @brief Allocates a new burst.
   *
   * @return Shared pointer to the allocated burst parameters.
   */
  inline std::shared_ptr<RivermaxBurst> allocate_burst() {
    auto burst = rx_bursts_mempool_->dequeue_burst();
    return burst;
  }

  /**
   * @brief Gets or allocates the current burst.
   *
   * This function checks if the current burst is null and allocates
   * a new one if necessary.
   * @return Shared pointer to the current burst parameters.
   */
  inline std::shared_ptr<RivermaxBurst> get_or_allocate_current_burst(int stream_id) {
    if (cur_out_burst_[stream_id] == nullptr) {
      cur_out_burst_[stream_id] = allocate_burst();
      if (cur_out_burst_[stream_id] == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
        return nullptr;
      }
      cur_out_burst_[stream_id]->reset_burst_with_updated_params(
          hds_on_, header_stride_size_, payload_stride_size_, gpu_direct_);
    }
    return cur_out_burst_[stream_id];
  }
  /**
   * @brief Enqueues the current burst and resets it.
   *
   * @return Status indicating the success or failure of the operation.
   */
  inline Status enqueue_and_reset_current_burst(int stream_id) {
    if (cur_out_burst_[stream_id] == nullptr) {
      HOLOSCAN_LOG_ERROR("Trying to enqueue an empty burst");
      return Status::NULL_PTR;
    }

    bool res = rx_bursts_out_queue_[stream_id]->enqueue_burst(cur_out_burst_[stream_id]);
    reset_current_burst(stream_id);
    if (!res) {
      HOLOSCAN_LOG_ERROR("Failed to enqueue burst");
      return Status::NO_SPACE_AVAILABLE;
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Resets the current burst.
   */
  inline void reset_current_burst(int stream_id) { cur_out_burst_[stream_id] = nullptr; }

 protected:
  bool send_packet_ext_info_ = false;
  int port_id_ = 0;
  int queue_id_ = 0;
  uint16_t burst_out_size_ = 0;
  int gpu_id_ = -1;
  bool hds_on_ = false;
  bool gpu_direct_ = false;
  size_t header_stride_size_ = 0;
  size_t payload_stride_size_ = 0;
  bool using_shared_out_queue_ = true;
  std::unique_ptr<IAnoBurstsCollection> rx_bursts_mempool_ = nullptr;
  std::unordered_map<int, std::shared_ptr<AnoBurstsQueue>> rx_bursts_out_queue_;
  std::unordered_map<int, std::shared_ptr<RivermaxBurst>> cur_out_burst_;
  AnoBurstExtendedInfo burst_info_;
  std::unique_ptr<RivermaxBurst::BurstHandler> burst_handler_;
};

};  // namespace holoscan::advanced_network

#endif /* BURST_MANAGER_H_ */
