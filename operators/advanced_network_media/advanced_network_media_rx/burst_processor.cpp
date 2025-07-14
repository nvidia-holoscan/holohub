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

#include "burst_processor.h"
#include "../common/adv_network_media_common.h"
#include "advanced_network/common.h"

namespace holoscan::ops {

BurstProcessor::BurstProcessor(std::shared_ptr<PacketsToFramesConverter> converter)
    : packets_to_frames_converter_(converter) {}

void BurstProcessor::process_burst(BurstParams* burst, bool hds_enabled) {
  if (!burst || burst->hdr.hdr.num_pkts == 0) { return; }

  ensure_converter_configuration(burst);
  process_packets_in_burst(burst, hds_enabled);
}

void BurstProcessor::ensure_converter_configuration(BurstParams* burst) {
  static bool burst_info_initialized = false;
  static size_t last_header_stride = 0;
  static size_t last_payload_stride = 0;
  static bool last_hds_on = false;
  static bool last_payload_on_cpu = false;

  // Access burst extended info from custom_burst_data
  const auto* burst_info =
      reinterpret_cast<const AnoBurstExtendedInfo*>(&(burst->hdr.custom_burst_data));

  if (!burst_info_initialized) {
    // First-time initialization
    packets_to_frames_converter_->configure_burst_parameters(
        burst_info->header_stride_size, burst_info->payload_stride_size, burst_info->hds_on);

    // Set source memory location based on burst info
    packets_to_frames_converter_->set_source_memory_location(burst_info->payload_on_cpu);

    last_header_stride = burst_info->header_stride_size;
    last_payload_stride = burst_info->payload_stride_size;
    last_hds_on = burst_info->hds_on;
    last_payload_on_cpu = burst_info->payload_on_cpu;
    burst_info_initialized = true;

    HOLOSCAN_LOG_INFO(
        "Burst configuration initialized: header_stride={}, payload_stride={}, hds_on={}, "
        "payload_on_cpu={}",
        burst_info->header_stride_size,
        burst_info->payload_stride_size,
        burst_info->hds_on,
        burst_info->payload_on_cpu);
  } else {
    // Check for configuration changes after strategy confirmation
    bool config_changed =
        (last_header_stride != burst_info->header_stride_size ||
         last_payload_stride != burst_info->payload_stride_size ||
         last_hds_on != burst_info->hds_on || last_payload_on_cpu != burst_info->payload_on_cpu);

    if (config_changed) {
      HOLOSCAN_LOG_WARN(
          "Burst configuration changed after strategy confirmation - forcing re-detection");
      HOLOSCAN_LOG_INFO(
          "Old config: header_stride={}, payload_stride={}, hds_on={}, payload_on_cpu={}",
          last_header_stride,
          last_payload_stride,
          last_hds_on,
          last_payload_on_cpu);
      HOLOSCAN_LOG_INFO(
          "New config: header_stride={}, payload_stride={}, hds_on={}, payload_on_cpu={}",
          burst_info->header_stride_size,
          burst_info->payload_stride_size,
          burst_info->hds_on,
          burst_info->payload_on_cpu);

      // Force strategy re-detection due to configuration change
      packets_to_frames_converter_->force_strategy_redetection();

      // Re-initialize with new configuration
      packets_to_frames_converter_->configure_burst_parameters(
          burst_info->header_stride_size, burst_info->payload_stride_size, burst_info->hds_on);

      packets_to_frames_converter_->set_source_memory_location(burst_info->payload_on_cpu);

      last_header_stride = burst_info->header_stride_size;
      last_payload_stride = burst_info->payload_stride_size;
      last_hds_on = burst_info->hds_on;
      last_payload_on_cpu = burst_info->payload_on_cpu;
    }
  }

  // Log strategy detection progress (only during initial detection phase)
  static bool strategy_logged = false;
  if (!strategy_logged &&
      packets_to_frames_converter_->get_current_strategy() != CopyStrategy::UNKNOWN) {
    HOLOSCAN_LOG_INFO(
        "Packet copy strategy detected: {}",
        packets_to_frames_converter_->get_current_strategy() == CopyStrategy::CONTIGUOUS
            ? "CONTIGUOUS"
            : "STRIDED");
    strategy_logged = true;
  }
}

void BurstProcessor::process_packets_in_burst(BurstParams* burst, bool hds_enabled) {
  constexpr int GPU_PKTS = 1;
  constexpr int CPU_PKTS = 0;

  for (size_t packet_index = 0; packet_index < burst->hdr.hdr.num_pkts; packet_index++) {
    const uint8_t* rtp_header =
        reinterpret_cast<const uint8_t*>(burst->pkts[CPU_PKTS][packet_index]);

    uint8_t* payload = hds_enabled
                           ? reinterpret_cast<uint8_t*>(burst->pkts[GPU_PKTS][packet_index])
                           : reinterpret_cast<uint8_t*>(burst->pkts[CPU_PKTS][packet_index]) +
                                 RTP_SINGLE_SRD_HEADER_SIZE;

    RtpParams rtp_params;
    if (!parse_rtp_header(rtp_header, rtp_params)) {
      HOLOSCAN_LOG_ERROR("Failed to parse RTP header for packet {}, skipping", packet_index);
      continue;
    }

    packets_to_frames_converter_->process_incoming_packet(rtp_params, payload);
  }
}

bool BurstProcessor::has_pending_copy() const {
  return packets_to_frames_converter_->has_pending_copy();
}

}  // namespace holoscan::ops
