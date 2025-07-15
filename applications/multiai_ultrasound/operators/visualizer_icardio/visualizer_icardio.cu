/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "visualizer_icardio.cuh"

namespace holoscan::ops {

__global__ void gen_coords_kernel(unsigned int count, int property_size, const float* input,
                                  float* output) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= count) { return; }

  output[index * property_size + 0] = input[index * 2 + 1];
  output[index * property_size + 1] = input[index * 2 + 0];
  if (property_size == 3) {
    // keypoint
    output[index * property_size + 2] = 0.01f;
  } else if (property_size == 4) {
    // key area
    output[index * property_size + 2] = 0.04f;
    output[index * property_size + 3] = 0.02f;
  }
}

void gen_coords(unsigned int offset, unsigned int count, int property_size, const float* input,
                float* output, cudaStream_t stream) {
  dim3 block(8, 1, 1);
  dim3 grid((count + block.x - 1) / block.x, 1, 1);

  gen_coords_kernel<<<grid, block, 0, stream>>>(count, property_size, input + 2 * offset, output);
  CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace holoscan::ops
