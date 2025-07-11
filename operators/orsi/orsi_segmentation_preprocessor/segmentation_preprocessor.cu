/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "segmentation_preprocessor.cuh"

namespace holoscan::ops::orsi {
namespace segmentation_preprocessor {
__forceinline__ __device__ uint32_t hwc_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (y * shape.width + x) * shape.channels + c;
}

__forceinline__ __device__ uint32_t nchw_to_index(Shape shape, uint32_t y, uint32_t x, uint32_t c) {
  return (c * shape.height + y) * shape.width + x;
}

template <enum DataFormat>
__forceinline__ __device__ uint32_t data_format_to_index(Shape shape, uint32_t y, uint32_t x,
                                                         uint32_t c) {}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kHWC>(Shape shape, uint32_t y,
                                                                           uint32_t x, uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template<>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNHWC>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return hwc_to_index(shape, y, x, c);
}

template <>
__forceinline__ __device__ uint32_t data_format_to_index<DataFormat::kNCHW>(Shape shape, uint32_t y,
                                                                            uint32_t x,
                                                                            uint32_t c) {
  return nchw_to_index(shape, y, x, c);
}

__forceinline__ __device__ uint32_t hw1_to_index(Shape shape, uint32_t y, uint32_t x) {
  return y * shape.width + x;
}

template <enum DataFormat data_format>
__global__ void preprocessing_kernel(Shape shape, const float* input, float* output, float* means,
                                                                                      float* stds) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= shape.width) || (y >= shape.height)) { return; }

  for (uint32_t c = 0; c < shape.channels; c++) {
    output[data_format_to_index<data_format>(shape, y, x, c)] =
        (input[data_format_to_index<data_format>(shape, y, x, c)] - means[c]) / stds[c];
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void cuda_preprocess(enum DataFormat data_format, Shape shape, const float* input, float* output,
                                                                      float* means, float* stds) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(shape.width, block.x), ceil_div(shape.height, block.y), 1);

  switch (data_format) {
    case DataFormat::kNCHW:
      preprocessing_kernel<DataFormat::kNCHW><<<grid, block>>>(shape, input, output, means, stds);
      break;
    case DataFormat::kHWC:
      preprocessing_kernel<DataFormat::kHWC><<<grid, block>>>(shape, input, output, means, stds);
      break;
    case DataFormat::kNHWC:
      preprocessing_kernel<DataFormat::kNHWC><<<grid, block>>>(shape, input, output,  means, stds);
      break;
      }
}

}  // namespace segmentation_preprocessor
}  // namespace holoscan::ops::orsi
