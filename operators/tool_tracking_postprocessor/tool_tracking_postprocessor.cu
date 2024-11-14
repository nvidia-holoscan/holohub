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

#include <algorithm>

#include "tool_tracking_postprocessor.cuh"

namespace holoscan::ops {

static __device__ __host__ uint32_t ceil_div(uint32_t numerator, uint32_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

__global__ void filter_binary_mask_kernel(uint32_t width, uint32_t height, uint32_t index,
                                          const float3* colors, const float* binary_mask,
                                          float4* colored_mask) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= width) || (y >= height)) { return; }

  float value = binary_mask[((index * height) + y) * width + x];

  const float minV = 0.3f;
  const float maxV = 0.99f;
  const float range = maxV - minV;
  value = min(max(value, minV), maxV);
  value -= minV;
  value /= range;
  value *= 0.7f;

  const float4 dst = colored_mask[y * width + x];
  colored_mask[y * width + x] = make_float4((1.0f - value) * dst.x + colors[index].x * value,
                                            (1.0f - value) * dst.y + colors[index].y * value,
                                            (1.0f - value) * dst.z + colors[index].z * value,
                                            (1.0f - value) * dst.w + 1.f * value);
}

__global__ void filter_coordinates_kernel(uint32_t count, float min_prob, const float* probs,
                                          const float2* scaled_coords,
                                          float3* filtered_scaled_coords, uint32_t width,
                                          uint32_t height, const float3* colors,
                                          const float* binary_mask, float4* colored_mask) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  // the third component of the coordinate is the size of the crosses and the text
  constexpr float ITEM_SIZE = 0.05f;

  // check if the probability meets the minimum probability
  if (probs[index] > min_prob) {
    filtered_scaled_coords[index] =
        make_float3(scaled_coords[index].x, scaled_coords[index].y, ITEM_SIZE);
    // add the binary mask to the result only if probabiliy is met
    const dim3 block(32, 32, 1);
    const dim3 grid(ceil_div(width, block.x), ceil_div(height, block.y), 1);
    filter_binary_mask_kernel<<<grid, block>>>(
        width, height, index, colors, binary_mask, colored_mask);
  } else {
    // move outside of the screen
    filtered_scaled_coords[index] = make_float3(-1.f, -1.f, ITEM_SIZE);
  }
}
void cuda_postprocess(uint32_t count, float min_prob, const float* probs,
                      const float2* scaled_coords, float3* filtered_scaled_coords, uint32_t width,
                      uint32_t height, const float3* colors, const float* binary_mask,
                      float4* colored_mask, cudaStream_t cuda_stream) {
  // initialize the output mask to zero
  CUDA_TRY(cudaMemsetAsync(colored_mask, 0, width * height * sizeof(float4), cuda_stream));

  const dim3 block(count, 1, 1);
  const dim3 grid(1, 1, 1);
  filter_coordinates_kernel<<<grid, block, 0, cuda_stream>>>(count,
                                                             min_prob,
                                                             probs,
                                                             scaled_coords,
                                                             filtered_scaled_coords,
                                                             width,
                                                             height,
                                                             colors,
                                                             binary_mask,
                                                             colored_mask);
  CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace holoscan::ops
