/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>

// CUDA kernel to convert 16-bit per channel images to 8-bit per channel images
__global__ void convert_16bit_to_8bit_kernel(const uint16_t* input, uint8_t* output, int width,
                                             int height, int input_channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int input_idx = (y * width + x) * input_channels;
  int output_idx = (y * width + x) * input_channels;  // Same number of channels

  // Convert 16-bit to 8-bit by right-shifting 8 bits
  // This effectively divides by 256, converting 16-bit range (0-65535) to 8-bit range (0-255)
  for (int c = 0; c < input_channels; c++) {
    output[output_idx + c] = static_cast<uint8_t>(input[input_idx + c] >> 8);
  }
}

// C++ wrapper function to launch the CUDA kernel
extern "C" void launch_convert_16bit_to_8bit_kernel(const uint16_t* input, uint8_t* output,
                                                    int width, int height, int input_channels) {
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);

  convert_16bit_to_8bit_kernel<<<grid_size, block_size>>>(
      input, output, width, height, input_channels);
}
