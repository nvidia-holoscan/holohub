/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tool_tracking_postprocessor.cuh"

namespace holoscan::ops {
namespace tool_tracking_postprocessor {

__global__ void postprocessing_kernel(uint32_t width, uint32_t height, const float3 color,
                                      bool first, const float* input, float4* output) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= width) || (y >= height)) { return; }

  float value = input[y * width + x];

  const float minV = 0.3f;
  const float maxV = 0.99f;
  const float range = maxV - minV;
  value = min(max(value, minV), maxV);
  value -= minV;
  value /= range;
  value *= 0.7f;

  const float4 dst = first ? make_float4(0.f, 0.f, 0.f, 0.f) : output[y * width + x];
  output[y * width + x] = make_float4((1.0f - value) * dst.x + color.x * value,
                                      (1.0f - value) * dst.y + color.y * value,
                                      (1.0f - value) * dst.z + color.z * value,
                                      (1.0f - value) * dst.w + 1.f * value);
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_postprocess(uint32_t width, uint32_t height, const std::array<float, 3>& color,
                      bool first, const float* input, float4* output, cudaStream_t cuda_stream) {
  const dim3 block(32, 32, 1);
  const dim3 grid(ceil_div(width, block.x), ceil_div(height, block.y), 1);
  postprocessing_kernel<<<grid, block, 0, cuda_stream>>>(
      width, height, make_float3(color[0], color[1], color[2]), first, input, output);
}

}  // namespace tool_tracking_postprocessor
}  // namespace holoscan::ops
