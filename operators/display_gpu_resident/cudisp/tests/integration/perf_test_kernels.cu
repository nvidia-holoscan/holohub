/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuda_runtime.h>

static __global__ void fill_flat_kernel(uint32_t* buf, uint32_t count, uint32_t color) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) buf[idx] = color;
}

void launch_fill_flat(uint32_t* buf, uint32_t count, uint32_t color, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((count + block.x - 1) / block.x);
  fill_flat_kernel<<<grid, block, 0, stream>>>(buf, count, color);
}
