/*
 * Copyright © 2017-2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
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
 * @brief Class responsible for handling burst operations.
 */
class BurstHandler {
 public:
  static constexpr int MAX_PKT_BURST = 9100;  ///< Maximum number of packets per burst.

  /**
   * @brief Constructs a BurstHandler object.
   *
   * @param send_packet_info Flag indicating whether to send packet info.
   * @param port_id The port ID.
   * @param queue_id The queue ID.
   * @param gpu_direct Flag indicating whether GPU direct is enabled.
   */
  BurstHandler(bool send_packet_info, int port_id, int queue_id, bool gpu_direct);

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
    ;
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
   * @brief Creates and initializes a new burst with the given burst ID
   *
   * @param burst_id The ID of the burst to create.
   * @return A shared pointer to the created burst.
   */
  std::shared_ptr<AdvNetBurstParams> create_burst(uint16_t burst_id);

  /**
   * @brief Deletes a burst and frees its associated resources.
   *
   * @param burst A shared pointer to the burst to delete.
   */
  void delete_burst(std::shared_ptr<AdvNetBurstParams> burst);

  /**
   * @brief Gets the ID of a burst.
   *
   * @param burst A reference to the burst.
   * @return The ID of the burst.
   */
  inline uint16_t get_burst_id(AdvNetBurstParams& burst) const {
    auto burst_info = get_burst_info(burst);
    if (burst_info == nullptr) { return 0; }
    return burst_info->burst_id;
  }

  /**
   * @brief Gets the tag of a burst.
   *
   * @param burst A reference to the burst.
   * @return The tag of the burst.
   */
  inline uint16_t get_burst_tag(AdvNetBurstParams& burst) const {
    auto burst_info = get_burst_info(burst);
    if (burst_info == nullptr) { return 0; }
    return burst_info->tag;
  }

  /**
   * @brief Gets the extended info of a burst.
   *
   * @param burst A reference to the burst.
   * @return A pointer to the extended info of the burst.
   */
  inline AnoBurstExtendedInfo* get_burst_info(AdvNetBurstParams& burst) const {
    return reinterpret_cast<AnoBurstExtendedInfo*>(&(burst.hdr.custom_burst_data));
  }

  /**
   * @brief Appends a packet to a burst.
   *
   * @param burst The burst to append the packet to.
   * @param packet_ind_in_out_burst The index of the packet in the burst.
   * @param packet_data The extended data of the packet.
   */
  inline void append_packet_to_burst(AdvNetBurstParams& burst, size_t packet_ind_in_out_burst,
                                     bool hds_on, const RmaxPacketData& packet_data) {
    RmaxPacketExtendedInfo* rx_packet_info = nullptr;

    if (m_send_packet_info) {
      rx_packet_info =
          reinterpret_cast<RmaxPacketExtendedInfo*>(burst.pkt_extra_info[packet_ind_in_out_burst]);
      rx_packet_info->timestamp = packet_data.extended_info.timestamp;
      rx_packet_info->flow_tag = packet_data.extended_info.flow_tag;
    }
    if (hds_on) {
      burst.pkts[0][packet_ind_in_out_burst] = packet_data.header_ptr;
      burst.pkts[1][packet_ind_in_out_burst] = packet_data.payload_ptr;
      burst.pkt_lens[0][packet_ind_in_out_burst] = packet_data.header_length;
      burst.pkt_lens[1][packet_ind_in_out_burst] = packet_data.payload_length;
    } else {
      burst.pkts[0][packet_ind_in_out_burst] = packet_data.payload_ptr;
      burst.pkts[1][packet_ind_in_out_burst] = nullptr;
      burst.pkt_lens[0][packet_ind_in_out_burst] = packet_data.header_length;
      burst.pkt_lens[1][packet_ind_in_out_burst] = 0;
    }
  }

  /**
   * @brief Updates the info of a burst.
   *
   * @param burst The burst to update.
   * @param hds_on Flag indicating whether HDS is on.
   * @param header_stride_size The size of the header stride.
   * @param payload_stride_size The size of the payload stride.
   * @param gpu_direct Flag indicating whether GPU direct is enabled.
   */
  inline void update_burst_info(AdvNetBurstParams& burst, bool hds_on, size_t header_stride_size,
                                size_t payload_stride_size, bool gpu_direct) {
    burst.hdr.hdr.q_id = m_queue_id;
    burst.hdr.hdr.port_id = m_port_id;

    auto burst_info = get_burst_info(burst);

    if (burst_info == nullptr) { return; }

    burst_info->hds_on = hds_on;
    burst_info->header_stride_size = header_stride_size;
    burst_info->payload_stride_size = payload_stride_size;
    burst_info->header_seg_idx = 0;
    burst_info->payload_on_cpu = !gpu_direct;

    if (hds_on) {
      burst_info->payload_seg_idx = 1;
      burst_info->header_on_cpu = true;
    } else {
      burst_info->payload_seg_idx = 0;
      burst_info->header_on_cpu = gpu_direct;
    }
  }

  /**
   * @brief Gets the number of packets in a burst.
   *
   * @param burst A reference to the burst.
   * @return The number of packets in the burst.
   */
  inline uint16_t get_num_packets(AdvNetBurstParams& burst) const { return burst.hdr.hdr.num_pkts; }

  /**
   * @brief Sets the number of packets in a burst.
   *
   *  @param burst A reference to the burst..
   * @param num_pkts The number of packets to set.
   */
  inline void set_num_packets(AdvNetBurstParams& burst, uint16_t num_pkts) {
    burst.hdr.hdr.num_pkts = num_pkts;
  }

 private:
  bool m_send_packet_info;            ///< Flag indicating whether to send packet info.
  int m_port_id;                      ///< The port ID.
  int m_queue_id;                     ///< The queue ID.
  bool m_gpu_direct;                  ///< Flag indicating whether GPU direct is enabled.
  AnoBurstExtendedInfo m_burst_info;  ///< The extended info of the burst.
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
   * @param send_packet_info Flag indicating whether to send packet info.
   * @param port_id ID of the port.
   * @param queue_id ID of the queue.
   * @param burst_out_size Size of the burst output.
   * @param gpu_id ID of the GPU.
   * @param rx_bursts_out_queue Shared pointer to the output queue for RX bursts.
   */
  RxBurstsManager(bool send_packet_info, int port_id, int queue_id, uint16_t burst_out_size = 0,
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
    // Consider to add check for pointers
    std::shared_ptr<AdvNetBurstParams> cur_burst = get_or_allocate_current_burst();
    if (cur_burst == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
      return ReturnStatus::no_free_chunks;
    }

    size_t packet_ind_in_out_burst = m_burst_handler->get_num_packets(*cur_burst);
    m_burst_handler->append_packet_to_burst(
        *cur_burst, packet_ind_in_out_burst, m_hds_on, packet_data);
    packet_ind_in_out_burst++;
    m_burst_handler->set_num_packets(*cur_burst, packet_ind_in_out_burst);

    // Enqueue the current burst if it meets the minimum size requirement
    if (packet_ind_in_out_burst >= m_burst_out_size) {
      bool res = m_rx_bursts_out_queue->enqueue_burst(cur_burst);
      reset_current_burst();
      if (!res) {
        HOLOSCAN_LOG_ERROR("Failed to enqueue burst");
        return ReturnStatus::failure;
      }
    }

    return ReturnStatus::success;
  }

  /**
   * @brief Gets an RX burst.
   *
   * @param burst Pointer to the burst parameters.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  inline ReturnStatus get_rx_burst(AdvNetBurstParams** burst) {
    *burst = m_rx_bursts_out_queue->dequeue_burst().get();
    if (*burst == nullptr) { return ReturnStatus::failure; }
    return ReturnStatus::success;
  }

  /**
   * @brief Marks the RX burst as done.
   *
   * @param burst Pointer to the burst parameters.
   */
  void rx_burst_done(AdvNetBurstParams* burst);

 protected:
  /**
   * @brief Allocates a new burst.
   *
   * @return Shared pointer to the allocated burst parameters.
   */
  inline std::shared_ptr<AdvNetBurstParams> allocate_burst() {
    auto burst = m_rx_bursts_mempool->dequeue_burst();
    if (burst != nullptr) { m_burst_handler->set_num_packets(*burst, 0); }
    return burst;
  }

  /**
   * @brief Gets or allocates the current burst.
   *
   * @return Shared pointer to the current burst parameters.
   */
  inline std::shared_ptr<AdvNetBurstParams> get_or_allocate_current_burst() {
    if (m_cur_out_burst == nullptr) {
      m_cur_out_burst = allocate_burst();
      if (m_cur_out_burst == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to allocate burst, running out of resources");
        return nullptr;
      }
      m_burst_handler->update_burst_info(
          *m_cur_out_burst, m_hds_on, m_header_stride_size, m_payload_stride_size, m_gpu_direct);
    }
    return m_cur_out_burst;
  }

  /**
   * @brief Resets the current burst.
   */
  inline void reset_current_burst() { m_cur_out_burst = nullptr; }

 protected:
  bool m_send_packet_info = false;
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
  std::shared_ptr<AdvNetBurstParams> m_cur_out_burst = nullptr;
  AnoBurstExtendedInfo m_burst_info;
  std::unique_ptr<BurstHandler> m_burst_handler;
};

};  // namespace holoscan::ops

#endif /* BURST_MANAGER_H_ */
