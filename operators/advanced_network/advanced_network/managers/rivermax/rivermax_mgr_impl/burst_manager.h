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
#include <chrono>
#include <atomic>
#include <string>

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
    if (burst_info == nullptr) {
      return 0;
    }
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

    if (burst_info == nullptr) {
      return;
    }

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
  inline BurstFlags get_burst_flags() const { return static_cast<BurstFlags>(hdr.hdr.burst_flags); }

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
  static constexpr uint32_t DEFAULT_NUM_RX_BURSTS = 256;
  static constexpr uint32_t GET_BURST_TIMEOUT_MS = 1000;

  // Pool capacity monitoring thresholds (as percentage of total pool size)
  // Default pool capacity thresholds (percentages) - can be overridden via configuration
  static constexpr uint32_t DEFAULT_POOL_LOW_CAPACITY_THRESHOLD_PERCENT = 25;  // Warning level
  static constexpr uint32_t DEFAULT_POOL_CRITICAL_CAPACITY_THRESHOLD_PERCENT =
      10;                                                                  // Start dropping
  static constexpr uint32_t DEFAULT_POOL_RECOVERY_THRESHOLD_PERCENT = 50;  // Stop dropping

  // Burst dropping policies (simplified)
  enum class BurstDropPolicy {
    NONE = 0,               // No dropping
    CRITICAL_THRESHOLD = 1  // Drop new bursts when critical, stop when recovered (default)
  };

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
  inline Status submit_next_packet(const RivermaxPacketData& packet_data) {
    get_or_allocate_current_burst();
    if (cur_out_burst_ == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
      return Status::NO_FREE_BURST_BUFFERS;
    }

    cur_out_burst_->append_packet(packet_data);

    if (cur_out_burst_->get_num_packets() >= burst_out_size_) {
      return enqueue_and_reset_current_burst();
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
  inline Status get_rx_burst(BurstParams** burst) {
    if (using_shared_out_queue_) {
      throw std::logic_error("Cannot get RX burst when using shared output queue");
    }

    auto out_burst = rx_bursts_out_queue_->dequeue_burst().get();
    *burst = static_cast<BurstParams*>(out_burst);
    if (*burst == nullptr) {
      return Status::NULL_PTR;
    }
    return Status::SUCCESS;
  }

  /**
   * @brief Marks the RX burst as done.
   *
   * @param burst Pointer to the burst parameters.
   */
  void rx_burst_done(RivermaxBurst* burst);

  /**
   * @brief Gets the current pool capacity utilization as a percentage.
   *
   * This monitors the MEMORY POOL where we allocate new bursts from.
   * Lower percentage = fewer available bursts = higher memory pressure.
   *
   * @return Pool utilization percentage (0-100).
   *         100% = all bursts available, 0% = no bursts available (pool exhausted)
   */
  inline uint32_t get_pool_utilization_percent() const {
    if (initial_pool_size_ == 0)
      return 0;
    size_t available = rx_bursts_mempool_->available_bursts();
    return static_cast<uint32_t>((available * 100) / initial_pool_size_);
  }

  /**
   * @brief Checks if pool capacity is below the specified threshold.
   *
   * @param threshold_percent Threshold percentage (0-100).
   * @return True if pool capacity is below threshold.
   */
  inline bool is_pool_capacity_below_threshold(uint32_t threshold_percent) const {
    return get_pool_utilization_percent() < threshold_percent;
  }

  /**
   * @brief Gets pool capacity status for monitoring.
   *
   * @return String description of current pool status.
   */
  std::string get_pool_status_string() const;

  /**
   * @brief Enables or disables adaptive burst dropping.
   *
   * @param enabled True to enable adaptive dropping.
   * @param policy Burst dropping policy to use.
   */
  inline void set_adaptive_burst_dropping(
      bool enabled, BurstDropPolicy policy = BurstDropPolicy::CRITICAL_THRESHOLD) {
    adaptive_dropping_enabled_ = enabled;
    burst_drop_policy_ = policy;
  }

  /**
   * @brief Configure pool capacity thresholds for adaptive dropping.
   *
   * @param low_threshold_percent Pool capacity % that triggers low capacity warnings (0-100)
   * @param critical_threshold_percent Pool capacity % that triggers burst dropping (0-100)
   * @param recovery_threshold_percent Pool capacity % that stops burst dropping (0-100)
   */
  inline void configure_pool_thresholds(uint32_t low_threshold_percent,
                                        uint32_t critical_threshold_percent,
                                        uint32_t recovery_threshold_percent) {
    pool_low_threshold_percent_ = low_threshold_percent;
    pool_critical_threshold_percent_ = critical_threshold_percent;
    pool_recovery_threshold_percent_ = recovery_threshold_percent;
  }

  /**
   * @brief Gets burst dropping statistics.
   *
   * @return String with burst dropping statistics.
   */
  std::string get_burst_drop_statistics() const;

 protected:
  /**
   * @brief Allocates a new burst.
   *
   * @return Shared pointer to the allocated burst parameters.
   */
  inline std::shared_ptr<RivermaxBurst> allocate_burst() {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto burst = rx_bursts_mempool_->dequeue_burst();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (burst != nullptr) {
      HOLOSCAN_LOG_DEBUG(
          "allocate_burst: dequeue_burst succeeded in {} μs (port_id: {}, queue_id: {}, burst_id: "
          "{})",
          duration.count(),
          port_id_,
          queue_id_,
          burst->get_burst_id());
    } else {
      HOLOSCAN_LOG_WARN(
          "allocate_burst: dequeue_burst FAILED (timeout/no bursts available) in {} μs (port_id: "
          "{}, queue_id: {})",
          duration.count(),
          port_id_,
          queue_id_);
    }

    return burst;
  }

  /**
   * @brief Gets or allocates the current burst.
   *
   * This function checks if the current burst is null and allocates
   * a new one if necessary.
   * @return Shared pointer to the current burst parameters.
   */
  inline std::shared_ptr<RivermaxBurst> get_or_allocate_current_burst() {
    if (cur_out_burst_ == nullptr) {
      cur_out_burst_ = allocate_burst();
      if (cur_out_burst_ == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
        return nullptr;
      }
      cur_out_burst_->reset_burst_with_updated_params(
          hds_on_, header_stride_size_, payload_stride_size_, gpu_direct_);
    }
    return cur_out_burst_;
  }
  /**
   * @brief Enqueues the current burst and resets it.
   *
   * @return Status indicating the success or failure of the operation.
   */
  inline Status enqueue_and_reset_current_burst() {
    if (cur_out_burst_ == nullptr) {
      HOLOSCAN_LOG_ERROR("Trying to enqueue an empty burst");
      return Status::NULL_PTR;
    }

    // Check if we should drop this COMPLETED burst due to critical pool capacity
    if (should_drop_burst_due_to_capacity()) {
      // Drop the completed burst by returning it to memory pool instead of enqueuing to output
      // queue (counter is already incremented inside should_drop_burst_due_to_capacity)
      rx_bursts_mempool_->enqueue_burst(cur_out_burst_);
      reset_current_burst();
      return Status::SUCCESS;
    }

    bool res = rx_bursts_out_queue_->enqueue_burst(cur_out_burst_);
    reset_current_burst();
    if (!res) {
      HOLOSCAN_LOG_ERROR("Failed to enqueue burst");
      return Status::NO_SPACE_AVAILABLE;
    }

    return Status::SUCCESS;
  }

  /**
   * @brief Resets the current burst.
   */
  inline void reset_current_burst() { cur_out_burst_ = nullptr; }

  /**
   * @brief Checks pool capacity and decides whether to drop bursts.
   *
   * @return True if burst should be dropped due to low capacity.
   */
  bool should_drop_burst_due_to_capacity();

  /**
   * @brief Logs pool capacity warnings and statistics.
   *
   * @param current_utilization Current pool utilization percentage.
   */
  void log_pool_capacity_status(uint32_t current_utilization) const;

  /**
   * @brief Implements generic adaptive burst dropping logic.
   *
   * @param current_utilization Current pool utilization percentage.
   * @return True if burst should be dropped based on network-level policies.
   */
  bool should_drop_burst_adaptive(uint32_t current_utilization) const;

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
  std::shared_ptr<IAnoBurstsCollection> rx_bursts_out_queue_ = nullptr;
  std::shared_ptr<RivermaxBurst> cur_out_burst_ = nullptr;
  AnoBurstExtendedInfo burst_info_;
  std::unique_ptr<RivermaxBurst::BurstHandler> burst_handler_;

  // Pool monitoring and adaptive dropping
  size_t initial_pool_size_ = DEFAULT_NUM_RX_BURSTS;
  bool adaptive_dropping_enabled_ = false;
  BurstDropPolicy burst_drop_policy_ = BurstDropPolicy::CRITICAL_THRESHOLD;

  // Configurable thresholds (defaults from constants)
  uint32_t pool_low_threshold_percent_ = DEFAULT_POOL_LOW_CAPACITY_THRESHOLD_PERCENT;
  uint32_t pool_critical_threshold_percent_ = DEFAULT_POOL_CRITICAL_CAPACITY_THRESHOLD_PERCENT;
  uint32_t pool_recovery_threshold_percent_ = DEFAULT_POOL_RECOVERY_THRESHOLD_PERCENT;

  // Critical threshold dropping state
  mutable bool in_critical_dropping_mode_ = false;  // Track if we're actively dropping

  // Statistics for burst dropping
  mutable std::atomic<uint64_t> total_bursts_dropped_{0};
  mutable std::atomic<uint64_t> bursts_dropped_low_capacity_{0};
  mutable std::atomic<uint64_t> bursts_dropped_critical_capacity_{0};
  mutable std::atomic<uint64_t> pool_capacity_warnings_{0};
  mutable std::atomic<uint64_t> pool_capacity_critical_events_{0};

  // Performance monitoring
  mutable std::chrono::steady_clock::time_point last_capacity_warning_time_;
  mutable std::chrono::steady_clock::time_point last_capacity_critical_time_;
};

};  // namespace holoscan::advanced_network

#endif /* BURST_MANAGER_H_ */
