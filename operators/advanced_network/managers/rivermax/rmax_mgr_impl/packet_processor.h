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

#ifndef PACKET_PROCESSOR_H_
#define PACKET_PROCESSOR_H_

#include <cstddef>
#include <iostream>
#include <memory>
#include <tuple>

#include "rmax_ano_data_types.h"
#include "burst_manager.h"
#include "adv_network_types.h"

namespace holoscan::ops {
using namespace ral::services;

/**
 * @brief Parameters for processing a chunk of packets.
 *
 * The PacketsChunkParams struct contains the parameters required for processing
 * a chunk of packets.
 */
struct PacketsChunkParams {
  uint8_t* header_ptr;
  uint8_t* payload_ptr;
  const ReceivePacketInfo* packet_info_array;
  size_t chunk_size;
  bool hds_on;
  size_t header_stride_size;
  size_t payload_stride_size;
};

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
   * This function processes the packets contained in the provided arrays and returns the
   * status along with the number of processed packets.
   *
   * @param params Struct containing packet processing parameters.
   * @return A tuple containing the status and the number of processed packets.
   */
  virtual std::tuple<ReturnStatus, size_t> process_packets(const PacketsChunkParams& params) = 0;
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
  explicit RxPacketProcessor(std::shared_ptr<RxBurstsManager> rx_burst_manager)
      : m_rx_burst_manager(rx_burst_manager) {
    if (m_rx_burst_manager == nullptr) {
      throw std::invalid_argument("RxPacketProcessor: rx_burst_manager is nullptr");
    }
  }

  /**
   * @brief Processes packets.
   *
   * This function processes the packets contained in the provided arrays and returns the
   * status along with the number of processed packets.
   *
   * @param params Struct containing packet processing parameters.
   * @return A tuple containing the status and the number of processed packets.
   */
  std::tuple<ReturnStatus, size_t> process_packets(const PacketsChunkParams& params) override {
    size_t processed_packets = 0;
    ReturnStatus status = ReturnStatus::success;

    if (params.chunk_size == 0) { return {status, processed_packets}; }

    auto remaining_packets = params.chunk_size;

    status = m_rx_burst_manager->set_next_chunk_params(
        params.chunk_size, params.hds_on, params.header_stride_size, params.payload_stride_size);

    if (status != ReturnStatus::success) { return {status, processed_packets}; }

    auto header_ptr = params.header_ptr;
    auto payload_ptr = params.payload_ptr;

    while (remaining_packets > 0) {
      RmaxPacketData rx_packet_data = {
          header_ptr,
          payload_ptr,
          params.packet_info_array[processed_packets].get_packet_sub_block_size(0),
          params.packet_info_array[processed_packets].get_packet_sub_block_size(1),
          {params.packet_info_array[processed_packets].get_packet_flow_tag(),
           params.packet_info_array[processed_packets].get_packet_timestamp()}};

      status = m_rx_burst_manager->submit_next_packet(rx_packet_data);

      if (status != ReturnStatus::success) { return {status, processed_packets}; }

      processed_packets++;
      remaining_packets--;
      header_ptr += params.header_stride_size;
      payload_ptr += params.payload_stride_size;
    }

    return {status, processed_packets};
  }

 private:
  std::shared_ptr<RxBurstsManager> m_rx_burst_manager;
};

};  // namespace holoscan::ops

#endif /* PACKET_PROCESSOR_H_ */
