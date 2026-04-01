/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuda_runtime.h>

static __global__ void render_bars_kernel(uint32_t* buf, uint32_t stride_px, uint32_t w,
                                          uint32_t h, uint32_t frame) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const uint32_t colors[] = {
    0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00,
    0xFFFF00FF, 0xFF00FFFF, 0xFFFFFFFF, 0xFF000000
  };

  uint32_t bar_w = w / 8;
  if (bar_w == 0) bar_w = 1;
  uint32_t shifted_x = (x + frame * 4) % w;
  uint32_t bar = shifted_x / bar_w;
  if (bar > 7) bar = 7;

  buf[y * stride_px + x] = colors[bar];
}

void launch_render_bars(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                        uint32_t frame, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  render_bars_kernel<<<grid, block, 0, stream>>>(buf, stride_px, w, h, frame);
}
