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

#ifndef RIVERMAX_CHUNK_CONSUMER_ANO_H_
#define RIVERMAX_CHUNK_CONSUMER_ANO_H_

#include <cstddef>
#include <iostream>

#include "rdk/core/core.h"
#include "rdk/services/services.h"

#include "rivermax_ano_data_types.h"
#include "packet_processor.h"
#include "advanced_network/types.h"

namespace holoscan::advanced_network {

using namespace rivermax::dev_kit::services;
using namespace rivermax::dev_kit::core;

/**
 * @brief Consumer class for handling Rivermax chunks and providing advanced_network bursts.
 *
 * The RivermaxChunkConsumerAno class acts as an adapter that consumes Rmax chunks
 * and produces advanced_network bursts. It processes the packets contained in the chunks,
 * updates the consumed and unconsumed byte counts, and manages the lifecycle
 * of the bursts. This class is designed to interface with the Rivermax framework
 * and provide the necessary functionality to handle and transform the data
 * into a format suitable for advanced_network to process.
 */
class RivermaxChunkConsumerAno : public IReceiveDataConsumer {
 public:
  /**
   * @brief Constructor for the RivermaxChunkConsumerAno class.
   *
   * Initializes the chunk consumer with the specified packet processor.
   *
   * @param packet_processor Shared pointer to the packet processor.
   */
  explicit RivermaxChunkConsumerAno(std::shared_ptr<RxPacketProcessor> packet_processor,
    size_t max_burst_size)
      : packet_processor_(std::move(packet_processor)),
        max_burst_size_(max_burst_size) {
    if (max_burst_size_ >= RivermaxBurst::MAX_PKT_IN_BURST) {
      max_burst_size_ = RivermaxBurst::MAX_PKT_IN_BURST;
    }
    packet_info_array_ = std::make_unique<ReceivePacketInfo[]>(RivermaxBurst::MAX_PKT_IN_BURST);
  }

  /**
   * @brief Destructor for the RivermaxChunkConsumerAno class.
   *
   * Ensures that all bursts are properly returned to the memory pool.
   */
  virtual ~RivermaxChunkConsumerAno() = default;

  ReturnStatus consume_chunk(const ReceiveChunk& chunk, const IReceiveStream& stream,
                             size_t& consumed_packets) override;

 protected:
  std::shared_ptr<RxPacketProcessor> packet_processor_;
  std::unique_ptr<ReceivePacketInfo[]> packet_info_array_;
  size_t max_burst_size_ = 0;
};

inline ReturnStatus RivermaxChunkConsumerAno::consume_chunk(const ReceiveChunk& chunk,
                                                            const IReceiveStream& stream,
                                                            size_t& consumed_packets) {
  consumed_packets = 0;
  if (packet_processor_ == nullptr) {
    HOLOSCAN_LOG_ERROR("Packet processor is not set");
    return ReturnStatus::failure;
  }

  auto chunk_size = chunk.get_length();
  if (chunk_size == 0) { return ReturnStatus::success; }
  if (chunk_size > max_burst_size_) {
    HOLOSCAN_LOG_WARN("Chunk size {} exceeds maximum burst size {}, discarding packets",
      chunk_size, max_burst_size_);
    chunk_size = max_burst_size_;
  }

  for (size_t i = 0; i < chunk_size; ++i) { packet_info_array_[i] = chunk.get_packet_info(i); }
  PacketsChunkParams params = {
      // header_ptr: Pointer to the header data
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(chunk.get_header_ptr())),
      // payload_ptr: Pointer to the payload data
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(chunk.get_payload_ptr())),
      // packet_info_array: Array of packet information
      packet_info_array_.get(),
      chunk_size,
      // hds_on: Header data splitting enabled
      chunk.is_header_data_split_on(),
      // header_stride_size: Stride size for the header data
      stream.get_header_stride_size(),
      // payload_stride_size: Stride size for the payload data
      stream.get_payload_stride_size(),
  };

  auto process_status = packet_processor_->process_packets(params, consumed_packets);
  if (process_status != Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Packet processing failed");
    return ReturnStatus::failure;
  }

  return ReturnStatus::success;
}

};  // namespace holoscan::advanced_network

#endif /* RIVERMAX_CHUNK_CONSUMER_ANO_H_ */
