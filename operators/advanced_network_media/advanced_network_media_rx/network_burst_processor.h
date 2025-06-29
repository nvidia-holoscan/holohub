/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_NETWORK_BURST_PROCESSOR_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_NETWORK_BURST_PROCESSOR_H_

#include <memory>
#include "holoscan/logger/logger.hpp"
#include "advanced_network/common.h"
#include "advanced_network/managers/rivermax/rivermax_ano_data_types.h"
#include "media_frame_assembler.h"
#include "../common/rtp_params.h"

namespace holoscan::ops {

// Targeted using declarations for specific types from advanced_network namespace
using holoscan::advanced_network::BurstParams;

/**
 * @brief Result structure for packet data extraction
 */
struct PacketExtractionResult {
  uint8_t* payload = nullptr;  ///< Pointer to packet payload data
  RtpParams rtp_params;        ///< Extracted RTP parameters
  bool success = false;        ///< Whether extraction was successful

  /// Implicit conversion to bool for easy error checking
  explicit operator bool() const { return success && payload != nullptr; }
};

/**
 * @brief Network burst processor that integrates with the frame assembler
 *
 * This class handles burst-level operations and forwards individual packets
 * to the MediaFrameAssembler for frame assembly processing.
 */
class NetworkBurstProcessor {
 public:
  /**
   * @brief Constructor
   * @param assembler The frame assembler with assembly controller
   */
  explicit NetworkBurstProcessor(std::shared_ptr<MediaFrameAssembler> assembler);

  /**
   * @brief Process a burst of packets
   * @param burst The burst containing packets to process
   * @note Assembler must be configured before calling this method
   */
  void process_burst(BurstParams* burst);

 private:
  /**
   * @brief Extract RTP header and payload from packet
   * @param burst The burst containing packets
   * @param packet_index Index of packet in burst
   * @return PacketExtractionResult containing payload pointer, RTP parameters, and success status
   */
  PacketExtractionResult extract_packet_data(BurstParams* burst, size_t packet_index);

 private:
  // Constants for packet array indexing
  static constexpr int CPU_PKTS = 0;
  static constexpr int GPU_PKTS = 1;

  std::shared_ptr<MediaFrameAssembler> assembler_;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_NETWORK_BURST_PROCESSOR_H_
