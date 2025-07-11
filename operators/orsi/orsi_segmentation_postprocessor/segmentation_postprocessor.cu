/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "segmentation_postprocessor.cuh"

namespace holoscan::ops::orsi {
namespace segmentation_postprocessor {


__forceinline__ __device__ uint32_t hw1_to_index(Shape shape, uint32_t y, uint32_t x) {
  return y * shape.width + x;
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}


__global__ void resizing_kernel(Shape input_shape, Shape output_shape, const uint8_t* input,
                              uint8_t* output, int32_t offset_x, int32_t offset_y) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= input_shape.width) || (y >= input_shape.height)) { return; }

  output[hw1_to_index(output_shape, y + offset_y, x  + offset_x)] =
                                                          input[hw1_to_index(input_shape, y, x)];
}

void cuda_resize(Shape input_shape, Shape output_shape, const uint8_t* input, uint8_t* output,
                                                           int32_t offset_x, int32_t offset_y) {
  dim3 block(32, 32, 1);
  dim3 grid(ceil_div(input_shape.width, block.x), ceil_div(input_shape.height, block.y), 1);

  resizing_kernel<<<grid, block>>>(input_shape, output_shape, input, output, offset_x, offset_y);
}

}  // namespace segmentation_postprocessor
}  // namespace holoscan::ops::orsi
