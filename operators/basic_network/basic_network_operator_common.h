/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <vector>
#include <stdexcept>

enum class L4Proto {
  TCP,
  UDP
};

/**
 * @brief Parameters for a burst of data to be sent or received over a network.
 *
 * @param data Pointer to the data to be sent or received.
 * @param len Length of the data (in bytes) to be sent or received.
 * @param num_pkts Number of packets in the burst.
 * @param packet_sizes Optional: Individual packet sizes. If empty, the expected behavior is that
 * the operator will use the max_payload_size parameter to determine the size of each packet.
 */
struct NetworkOpBurstParams {
  NetworkOpBurstParams(uint8_t *data, uint32_t len, uint32_t num_pkts) :
    data(data), len(len), num_pkts(num_pkts) {}

  // Constructor with packet sizes
  NetworkOpBurstParams(uint8_t *data, uint32_t len, uint32_t num_pkts,
                       const std::vector<uint32_t>& packet_sizes) :
    data(data), len(len), num_pkts(num_pkts), packet_sizes(packet_sizes) {
      if (num_pkts != packet_sizes.size()) {
        throw std::runtime_error("Number of packets does not match number of packet sizes");
      }
    }

  uint8_t *data;
  uint32_t len;
  uint32_t num_pkts;

  // Optional: individual packet sizes. If empty, use max_payload_size behavior
  std::vector<uint32_t> packet_sizes;
};
