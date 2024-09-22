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
  RmaxChunkConsumerAno(std::shared_ptr<RxPacketProcessor> packet_processor)
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
   * This function processes the packets contained in the provided chunk and updates the
   * consumed and unconsumed byte counts.
   *
   * @param chunk Reference to the IPOReceiveChunk containing the packets.
   * @param stream Reference to the IPOReceiveStream associated with the chunk.
   * @param consumed_packets Reference to the size_t that will be updated with the number of
   * consumed packets.
   * @param unconsumed_packets Reference to the size_t that will be updated with the number of
   * unconsumed packets.
   * @return ReturnStatus indicating the success or failure of the operation.
   */
  ReturnStatus consume_chunk_packets(IPOReceiveChunk& chunk, IPOReceiveStream& stream,
                                     size_t& consumed_packets, size_t& unconsumed_packets) override;

 protected:
  std::shared_ptr<RxPacketProcessor> m_packet_processor;
};

/**
 * @brief Consumes and processes packets from a given chunk.
 *
 * This function processes the packets contained in the provided chunk and updates the
 * consumed and unconsumed byte counts.
 *
 * No need to set initial values for consumed_packets and unconsumed_packets.
 *
 * @param chunk Reference to the IPOReceiveChunk containing the packets.
 * @param stream Reference to the IPOReceiveStream associated with the chunk.
 * @param consumed_packets Reference to the size_t that will be updated with the number of consumed
 * packets.
 * @param unconsumed_packets Reference to the size_t that will be updated with the number of
 * unconsumed packets.
 * @return ReturnStatus indicating the success or failure of the operation.
 */
inline ReturnStatus RmaxChunkConsumerAno::consume_chunk_packets(IPOReceiveChunk& chunk,
                                                                IPOReceiveStream& stream,
                                                                size_t& consumed_packets,
                                                                size_t& unconsumed_packets) {
  if (m_packet_processor == nullptr) {
    HOLOSCAN_LOG_ERROR("Packet processor is not set");
    return ReturnStatus::failure;
  }

  // Get pointers to the packet information, payload, and header data
  auto packet_info_array = chunk.get_completion_info_ptr();
  auto* payload_ptr = reinterpret_cast<uint8_t*>(chunk.get_completion_payload_ptr());
  auto* header_ptr = reinterpret_cast<uint8_t*>(chunk.get_completion_header_ptr());

  // Get the size of the chunk
  const auto chunk_size = chunk.get_completion_chunk_size();

  // Determine if header data splitting (HDS) is enabled
  const bool hds_on = (chunk_size > 0) ? (chunk.get_packet_header_size(0) > 0) : false;

  // Get the stride sizes for the header and payload data
  const auto header_stride_size = stream.get_header_stride_size();
  const auto payload_stride_size = stream.get_payload_stride_size();

  consumed_packets = 0;

  // Return success if there are no packets to process
  if (chunk_size == 0) {
    unconsumed_packets = 0;
    return ReturnStatus::success;
  }

  size_t processed_packets = 0;
  unconsumed_packets = chunk_size;

  auto status = m_packet_processor->process_packets(header_ptr,
                                                    payload_ptr,
                                                    packet_info_array,
                                                    chunk_size,
                                                    hds_on,
                                                    header_stride_size,
                                                    payload_stride_size,
                                                    processed_packets);

  consumed_packets += processed_packets;
  unconsumed_packets -= processed_packets;

  return status;
}

};  // namespace holoscan::ops

#endif /* RMAX_CHUNK_CONSUMER_ANO_H_ */