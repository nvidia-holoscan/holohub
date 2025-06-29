/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "network_burst_processor.h"
#include "../common/adv_network_media_common.h"
#include "advanced_network/common.h"

namespace holoscan::ops {

namespace ano = holoscan::advanced_network;
using holoscan::advanced_network::AnoBurstExtendedInfo;
using holoscan::advanced_network::BurstParams;

NetworkBurstProcessor::NetworkBurstProcessor(
    std::shared_ptr<MediaFrameAssembler> assembler)
    : assembler_(assembler) {
  if (!assembler_) {
    throw std::invalid_argument("MediaFrameAssembler cannot be null");
  }
}

void NetworkBurstProcessor::process_burst(BurstParams* burst) {
  if (!burst || burst->hdr.hdr.num_pkts == 0) {
    return;
  }

  // Process each packet through the frame assembler
  for (size_t i = 0; i < burst->hdr.hdr.num_pkts; ++i) {
    auto extraction_result = extract_packet_data(burst, i);

    // Skip packet if extraction failed
    if (!extraction_result) {
      ANM_LOG_WARN("Failed to extract payload from packet {}", i);
      continue;
    }

    ANM_PACKET_TRACE("About to process packet {}/{}: seq={}, m_bit={}, size={}, payload_ptr={}",
                     i + 1,
                     burst->hdr.hdr.num_pkts,
                     extraction_result.rtp_params.sequence_number,
                     extraction_result.rtp_params.m_bit,
                     extraction_result.rtp_params.payload_size,
                     static_cast<void*>(extraction_result.payload));

    assembler_->process_incoming_packet(extraction_result.rtp_params, extraction_result.payload);

    ANM_PACKET_TRACE("Processed packet {}/{}: seq={}, m_bit={}, size={}",
                     i + 1,
                     burst->hdr.hdr.num_pkts,
                     extraction_result.rtp_params.sequence_number,
                     extraction_result.rtp_params.m_bit,
                     extraction_result.rtp_params.payload_size);
  }
}

PacketExtractionResult NetworkBurstProcessor::extract_packet_data(BurstParams* burst,
                                                                  size_t packet_index) {
  PacketExtractionResult result;

  if (packet_index >= burst->hdr.hdr.num_pkts) {
    ANM_LOG_ERROR(
        "Packet index {} out of range (max: {})", packet_index, burst->hdr.hdr.num_pkts);
    return result;  // success = false, payload = nullptr
  }

  // Get HDS configuration from burst data
  const auto* burst_info =
      reinterpret_cast<const AnoBurstExtendedInfo*>(&(burst->hdr.custom_burst_data));

  if (burst_info->hds_on) {
    // Header-Data Split mode: headers on CPU, payloads on GPU
    uint8_t* header_ptr = reinterpret_cast<uint8_t*>(burst->pkts[CPU_PKTS][packet_index]);
    uint8_t* payload_ptr = reinterpret_cast<uint8_t*>(burst->pkts[GPU_PKTS][packet_index]);

    if (!header_ptr || !payload_ptr) {
      ANM_LOG_ERROR("Null pointer in HDS packet {}: header={}, payload={}",
                         packet_index,
                         static_cast<void*>(header_ptr),
                         static_cast<void*>(payload_ptr));
      return result;  // success = false, payload = nullptr
    }

    // Parse RTP header from CPU memory
    if (!parse_rtp_header(header_ptr, result.rtp_params)) {
      ANM_LOG_ERROR("Failed to parse RTP header for packet {}", packet_index);
      return result;  // success = false, payload = nullptr
    }

    ANM_PACKET_TRACE("HDS packet {}: header_ptr={}, payload_ptr={}, seq={}",
                     packet_index,
                     static_cast<void*>(header_ptr),
                     static_cast<void*>(payload_ptr),
                     result.rtp_params.sequence_number);

    result.payload = payload_ptr;
    result.success = true;
    return result;

  } else {
    // Standard mode: complete packet on CPU
    uint8_t* packet_ptr = reinterpret_cast<uint8_t*>(burst->pkts[CPU_PKTS][packet_index]);

    if (!packet_ptr) {
      ANM_LOG_ERROR("Null packet pointer for packet {}", packet_index);
      return result;  // success = false, payload = nullptr
    }

    // Parse RTP header from beginning of packet
    if (!parse_rtp_header(packet_ptr, result.rtp_params)) {
      ANM_LOG_ERROR("Failed to parse RTP header for packet {}", packet_index);
      return result;  // success = false, payload = nullptr
    }

    // Payload starts after RTP header
    uint8_t* payload_ptr = packet_ptr + RTP_SINGLE_SRD_HEADER_SIZE;

    ANM_PACKET_TRACE("Standard packet {}: packet_ptr={}, payload_ptr={}, seq={}",
                     packet_index,
                     static_cast<void*>(packet_ptr),
                     static_cast<void*>(payload_ptr),
                     result.rtp_params.sequence_number);

    result.payload = payload_ptr;
    result.success = true;
    return result;
  }
}

}  // namespace holoscan::ops
