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

#ifndef RMAX_CHUNK_CONSUMER_ANO_H_
#define RMAX_CHUNK_CONSUMER_ANO_H_

#include <cstddef>
#include <iostream>

#include "rmax_ano_data_types.h"
#include "rmax_service/ipo_chunk_consumer_base.h"
#include "rmax_service/rmax_ipo_receiver_service.h"
#include "packet_processor.h"
#include "adv_network_types.h"

namespace holoscan::ops {
using namespace ral::services;

/**
 * @brief Consumer class for handling Rmax chunks and providing ANO bursts.
 *
 * The RmaxChunkConsumerAno class acts as an adapter that consumes Rmax chunks
 * and produces ANO bursts. It processes the packets contained in the chunks,
 * updates the consumed and unconsumed byte counts, and manages the lifecycle
 * of the bursts. This class is designed to interface with the Rmax framework
 * and provide the necessary functionality to handle and transform the data
 * into a format suitable for ANO processing.
 */
class RmaxChunkConsumerAno : public IIPOChunkConsumer {
 public:
  /**
   * @brief Constructor for the RmaxChunkConsumerAno class.
   *
   * Initializes the chunk consumer with the specified packet processor.
   *
   * @param packet_processor Shared pointer to the packet processor.
   */
  explicit RmaxChunkConsumerAno(std::shared_ptr<RxPacketProcessor> packet_processor)
      : m_packet_processor(packet_processor) {}

  /**
   * @brief Destructor for the RmaxChunkConsumerAno class.
   *
   * Ensures that all bursts are properly returned to the memory pool.
   */
  virtual ~RmaxChunkConsumerAno() = default;

  /**
   * @brief Consumes and processes packets from a given chunk.
   *
   * This function processes the packets contained in the provided chunk and returns a tuple
   * containing the return status, the number of consumed packets, and the number of unconsumed
   * packets.
   *
   * @param chunk Reference to the IPOReceiveChunk containing the packets.
   * @param stream Reference to the IPOReceiveStream associated with the chunk.
   * @return std::tuple<ReturnStatus, size_t, size_t> containing the return status, the number of
   * consumed packets, and the number of unconsumed packets.
   */
  std::tuple<ReturnStatus, size_t, size_t> consume_chunk_packets(IPOReceiveChunk& chunk,
                                                                 IPOReceiveStream& stream) override;

 protected:
  std::shared_ptr<RxPacketProcessor> m_packet_processor;
};

/**
 * @brief Consumes and processes packets from a given chunk.
 *
 * This function processes the packets contained in the provided chunk and returns a tuple
 * containing the return status, the number of consumed packets, and the number of unconsumed
 * packets.
 *
 * @param chunk Reference to the IPOReceiveChunk containing the packets.
 * @param stream Reference to the IPOReceiveStream associated with the chunk.
 * @return std::tuple<ReturnStatus, size_t, size_t> containing the return status, the number of
 * consumed packets, and the number of unconsumed packets.
 */
inline std::tuple<ReturnStatus, size_t, size_t> RmaxChunkConsumerAno::consume_chunk_packets(
    IPOReceiveChunk& chunk, IPOReceiveStream& stream) {
  if (m_packet_processor == nullptr) {
    HOLOSCAN_LOG_ERROR("Packet processor is not set");
    return {ReturnStatus::failure, 0, 0};
  }

  const auto chunk_size = chunk.get_completion_chunk_size();
  if (chunk_size == 0) { return {ReturnStatus::success, 0, 0}; }

  PacketsChunkParams params = {
      // header_ptr: Pointer to the header data
      reinterpret_cast<uint8_t*>(chunk.get_completion_header_ptr()),
      // payload_ptr: Pointer to the payload data
      reinterpret_cast<uint8_t*>(chunk.get_completion_payload_ptr()),
      // packet_info_array: Array of packet information
      chunk.get_completion_info_ptr(),
      chunk_size,
      // hds_on: Header data splitting enabled
      (chunk_size > 0) ? (chunk.get_packet_header_size(0) > 0) : false,
      // header_stride_size: Stride size for the header data
      stream.get_header_stride_size(),
      // payload_stride_size: Stride size for the payload data
      stream.get_payload_stride_size(),
  };

  auto [status, processed_packets] = m_packet_processor->process_packets(params);

  return {status, processed_packets, chunk_size - processed_packets};
}

};  // namespace holoscan::ops

#endif /* RMAX_CHUNK_CONSUMER_ANO_H_ */
