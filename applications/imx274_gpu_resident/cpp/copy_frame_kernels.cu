/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "copy_frame_kernels.hpp"

__global__ void copy_frame_kernel(void* dst, unsigned char** src, unsigned int n_bytes) {
  const unsigned char* src_addr = *src;
  unsigned char* dst_addr = static_cast<unsigned char*>(dst);
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;
  const uint4* src4 = reinterpret_cast<const uint4*>(src_addr);
  uint4* dst4 = reinterpret_cast<uint4*>(dst_addr);
  unsigned int n_uint4 = n_bytes / sizeof(uint4);
  for (unsigned int i = tid; i < n_uint4; i += stride) {
    dst4[i] = src4[i];
  }

  // Copy any trailing bytes that do not fit into a uint4 chunk.
  unsigned int remainder_offset = n_uint4 * sizeof(uint4);
  for (unsigned int i = remainder_offset + tid; i < n_bytes; i += stride) {
    dst_addr[i] = src_addr[i];
  }
}

void launch_copy_frame(cudaStream_t stream, void* dst, unsigned char** src, unsigned int n_bytes) {
  constexpr unsigned int block_size = 256;
  // Size the grid for the vector path, but keep at least one block for small remainders.
  unsigned int n_copy_units = (n_bytes + sizeof(uint4) - 1) / sizeof(uint4);
  unsigned int grid_size =
      n_copy_units == 0 ? 1 : (n_copy_units + block_size - 1) / block_size;
  copy_frame_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, n_bytes);
}
