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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_BURST_PROCESSOR_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_BURST_PROCESSOR_H_

#include <memory>
#include "advanced_network/common.h"
#include "advanced_network/managers/rivermax/rivermax_ano_data_types.h"
#include "packets_to_frames_converter.h"

namespace holoscan::ops {

/**
 * @brief Processor for handling burst-level operations and packet extraction
 *
 * This class encapsulates burst configuration parsing, packet extraction,
 * and coordination with the PacketsToFramesConverter. It separates the
 * low-level packet processing from the high-level network management.
 *
 * Responsibilities:
 * - Parse burst configuration from AnoBurstExtendedInfo
 * - Configure PacketsToFramesConverter with burst parameters
 * - Extract RTP headers and payloads from burst packets
 * - Handle Header Data Split (HDS) configuration
 * - Detect and handle burst configuration changes
 */
class BurstProcessor {
 public:
  /**
   * @brief Constructor
   * @param converter Shared pointer to the packets-to-frames converter
   */
  explicit BurstProcessor(std::shared_ptr<PacketsToFramesConverter> converter);

  /**
   * @brief Process an entire burst of packets
   * @param burst The burst containing packets and configuration
   * @param hds_enabled Whether header data splitting is enabled from operator config
   */
  void process_burst(BurstParams* burst, bool hds_enabled);

  /**
   * @brief Check if there is accumulated packet data waiting to be copied
   * @return True if there is pending copy data, false otherwise
   */
  bool has_pending_copy() const;

 private:
  /**
   * @brief Ensure converter has proper configuration (configure once, validate on subsequent calls)
   * @param burst The burst containing configuration information
   */
  void ensure_converter_configuration(BurstParams* burst);

  /**
   * @brief Process all packets within a burst
   * @param burst The burst containing packets to process
   * @param hds_enabled Whether header data splitting is enabled
   */
  void process_packets_in_burst(BurstParams* burst, bool hds_enabled);

 private:
  std::shared_ptr<PacketsToFramesConverter> packets_to_frames_converter_;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_BURST_PROCESSOR_H_
