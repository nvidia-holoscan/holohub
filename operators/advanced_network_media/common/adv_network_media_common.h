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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_COMMON_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_COMMON_H_

#include <cstdint>
#include <stddef.h>
#include <holoscan/holoscan.hpp>
#include "rtp_params.h"

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("CUDA operation failed");                                \
    }                                                                                   \
  }

// Packet tracing debug flag - uncomment to enable extensive packet debugging
// #define ENABLE_PACKET_TRACING

#ifdef ENABLE_PACKET_TRACING
#define PACKET_TRACE_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[PACKET_TRACE] " fmt, ##__VA_ARGS__)
#else
#define PACKET_TRACE_LOG(fmt, ...) \
  do {                             \
  } while (0)
#endif

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_COMMON_H_
