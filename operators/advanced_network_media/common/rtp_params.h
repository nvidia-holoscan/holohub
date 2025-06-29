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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_RTP_PARAMS_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_RTP_PARAMS_H_

#include <cstdint>
#include <arpa/inet.h>
#include "rdk/services/services.h"

// Structure to hold parsed RTP parameters
struct RtpParams {
  uint32_t sequence_number;
  uint32_t timestamp;
  bool m_bit;        // Marker bit - indicates end of frame
  bool f_bit;        // Field bit
  uint16_t payload_size;  // Payload size from SRD field

  RtpParams() : sequence_number(0), timestamp(0), m_bit(false), f_bit(false), payload_size(0) {}

  /**
   * @brief Parse RTP header and populate this struct with extracted parameters
   * @param rtp_hdr Pointer to RTP header data
   * @return True if parsing successful, false if invalid RTP header
   */
  bool parse(const uint8_t* rtp_hdr) {
    // Validate RTP version (must be version 2)
    if ((rtp_hdr[0] & 0xC0) != 0x80) {
      return false;
    }

    // Extract CSRC count and calculate offset
    uint8_t cc = 0x0F & rtp_hdr[0];
    uint8_t offset = cc * RTP_HEADER_CSRC_GRANULARITY_BYTES;

    // Extract sequence number (16-bit + 16-bit extended)
    sequence_number = rtp_hdr[3] | rtp_hdr[2] << 8;
    sequence_number |= rtp_hdr[offset + 12] << 24 | rtp_hdr[offset + 13] << 16;

    // Extract field bit
    f_bit = !!(rtp_hdr[offset + 16] & 0x80);

    // Extract timestamp (32-bit, network byte order)
    timestamp = ntohl(*(uint32_t *)(rtp_hdr + 4));

    // Extract marker bit
    m_bit = !!(rtp_hdr[1] & 0x80);

    // Extract payload size from SRD Length field (2 bytes, network byte order)
    payload_size = (rtp_hdr[offset + 14] << 8) | rtp_hdr[offset + 15];

    return true;
  }
};

/**
 * @brief Parse RTP header and extract all parameters
 * @param rtp_hdr Pointer to RTP header data
 * @param params Reference to RtpParams struct to populate
 * @return True if parsing successful, false if invalid RTP header
 */
inline bool parse_rtp_header(const uint8_t* rtp_hdr, RtpParams& params) {
  return params.parse(rtp_hdr);
}
#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_RTP_PARAMS_H_
