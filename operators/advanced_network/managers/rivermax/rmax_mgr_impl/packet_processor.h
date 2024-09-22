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

#ifndef PACKET_PROCESSOR_H_
#define PACKET_PROCESSOR_H_

#include <cstddef>
#include <iostream>
#include <memory>

#include "rmax_ano_data_types.h"
#include "burst_manager.h"
#include "adv_network_types.h"

namespace holoscan::ops {
using namespace ral::services;

/**
 * @brief Interface for packet processors.
 *
 * The IPacketProcessor class defines the interface for processing packets.
 * Implementations of this interface are responsible for handling the packet
 * processing logic.
 */
class IPacketProcessor {
 public:
  /**
   * @brief Virtual destructor for the IPacketProcessor class.
   */
  virtual ~IPacketProcessor() = default;

  /**
   * @brief Processes packets.
   *
   * This function processes the packets contained in the provided arrays and updates the
   * processed packet count.
   *
   * @param header_ptr Pointer to the header data.
   * @param payload_ptr Pointer to the payload data.
   * @param packet_info_array Array of packet information.
   * @param chunk_size Size of the chunk to process.
   * @param hds_on Flag indicating if header data splitting (HDS) is enabled.
   * @param header_stride_size Stride size for the header data.
   * @param payload_stride_size Stride size for the payload data.
   * @param processed_packets Reference to the size_t that will be updated with the number of
   * processed packets.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  virtual ReturnStatus process_packets(uint8_t* header_ptr, uint8_t* payload_ptr,
                                       const ReceivePacketInfo* packet_info_array,
                                       size_t chunk_size, bool hds_on, size_t header_stride_size,
                                       size_t payload_stride_size, size_t& processed_packets) = 0;
};

/**
 * @brief Implementation of the IPacketProcessor interface for RX packets.
 *
 * The RxPacketProcessor class provides the implementation for processing RX packets.
 * It manages the lifecycle of the bursts and processes the packets contained in the chunks.
 */
class RxPacketProcessor : public IPacketProcessor {
 public:
  /**
   * @brief Constructor for the RxPacketProcessor class.
   *
   * Initializes the packet processor with the specified burst manager.
   *
   * @param rx_burst_manager Shared pointer to the burst manager.
   */
  RxPacketProcessor(std::shared_ptr<RxBurstsManager> rx_burst_manager)
      : m_rx_burst_manager(rx_burst_manager) {
    if (m_rx_burst_manager == nullptr) {
      throw std::invalid_argument("RxPacketProcessor: rx_burst_manager is nullptr");
    }
  }

  /**
   * @brief Processes packets.
   *
   * This function processes the packets contained in the provided arrays and updates the
   * processed packet count.
   *
   * @param header_ptr Pointer to the header data.
   * @param payload_ptr Pointer to the payload data.
   * @param packet_info_array Array of packet information.
   * @param chunk_size Size of the chunk to process.
   * @param hds_on Flag indicating if header data splitting (HDS) is enabled.
   * @param header_stride_size Stride size for the header data.
   * @param payload_stride_size Stride size for the payload data.
   * @param processed_packets Reference to the size_t that will be updated with the number of
   * processed packets.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  ReturnStatus process_packets(uint8_t* header_ptr, uint8_t* payload_ptr,
                               const ReceivePacketInfo* packet_info_array, size_t chunk_size,
                               bool hds_on, size_t header_stride_size, size_t payload_stride_size,
                               size_t& processed_packets) override {
    processed_packets = 0;

    // Return success if there are no packets to process
    if (chunk_size == 0) { return ReturnStatus::success; }

    auto remaining_packets = chunk_size;
    size_t consumed_packets_single_burst = 0;

    // Inform manager about a new chunk and it's params
    auto status = m_rx_burst_manager->set_next_chunk_params(
        chunk_size, hds_on, header_stride_size, payload_stride_size);

    if (status != ReturnStatus::success) { return status; }

    // Process packets one by one until all packets are processed
    while (remaining_packets > 0) {
      auto status =
          process_single_packet(header_ptr, payload_ptr, packet_info_array[processed_packets]);

      if (status != ReturnStatus::success) { return status; }

      processed_packets++;
      remaining_packets--;
      header_ptr += header_stride_size;
      payload_ptr += payload_stride_size;
    }

    return ReturnStatus::success;
  }

 private:
  std::shared_ptr<RxBurstsManager> m_rx_burst_manager;

  /**
   * @brief Processes a single packet.
   *
   * This function processes a single packet and submits it to the burst manager.
   *
   * @param header_ptr Pointer to the header data.
   * @param payload_ptr Pointer to the payload data.
   * @param packet_info Reference to the packet information.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  ReturnStatus process_single_packet(uint8_t* header_ptr, uint8_t* payload_ptr,
                                     const ReceivePacketInfo& packet_info) {
    RmaxPacketExtendedInfo rx_packet_info = {packet_info.get_packet_flow_tag(),
                                             packet_info.get_packet_timestamp()};

    return m_rx_burst_manager->submit_next_packet(header_ptr,
                                                  payload_ptr,
                                                  packet_info.get_packet_sub_block_size(0),
                                                  packet_info.get_packet_sub_block_size(1),
                                                  rx_packet_info);
  }
};

};  // namespace holoscan::ops

#endif /* PACKET_PROCESSOR_H_ */