/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>

#include <holoscan/logger/logger.hpp>

#define CUDA_TRY(stmt)                                                                        \
  {                                                                                           \
    cudaError_t cuda_status = stmt;                                                           \
    if (cudaSuccess != cuda_status) {                                                         \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({}).", \
                         #stmt,                                                               \
                         __LINE__,                                                            \
                         __FILE__,                                                            \
                         cudaGetErrorString(cuda_status),                                     \
                         static_cast<int>(cuda_status));                                      \
      throw std::runtime_error("CUDA runtime call failed");                                   \
    }                                                                                         \
  }

namespace holoscan::ops {

void cuda_postprocess(uint32_t count, float min_prob, const float* probs,
                      const float2* scaled_coords, float3* filtered_scaled_coords, uint32_t width,
                      uint32_t height, const float3* colors, const float* binary_mask,
                      float4* colored_mask, cudaStream_t cuda_stream);

}  // namespace holoscan::ops
