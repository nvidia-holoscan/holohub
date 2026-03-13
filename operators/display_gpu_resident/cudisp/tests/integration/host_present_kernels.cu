/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuda_runtime.h>

static __global__ void fill_bars_kernel(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                                        uint32_t frame) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const uint32_t colors[] = {
    0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00,
    0xFFFF00FF, 0xFF00FFFF, 0xFFFFFFFF, 0xFF000000
  };
  uint32_t bar = ((x + frame) % w) / (w / 8 + 1);
  if (bar > 7) bar = 7;
  buf[y * stride_px + x] = colors[bar];
}

static __global__ void fill_solid_kernel(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                                         uint32_t color) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  buf[y * stride_px + x] = color;
}

static __global__ void fill_gradient_kernel(uint32_t* buf, uint32_t stride_px, uint32_t w,
                                            uint32_t h, uint32_t frame) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  uint8_t r = static_cast<uint8_t>((x + frame) * 255 / w);
  uint8_t g = static_cast<uint8_t>(y * 255 / h);
  uint8_t b = static_cast<uint8_t>(128);
  buf[y * stride_px + x] = (0xFFu << 24) | (r << 16) | (g << 8) | b;
}

void launch_fill_bars(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                      uint32_t frame, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  fill_bars_kernel<<<grid, block, 0, stream>>>(buf, stride_px, w, h, frame);
}

void launch_fill_solid(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                       uint32_t color, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  fill_solid_kernel<<<grid, block, 0, stream>>>(buf, stride_px, w, h, color);
}

void launch_fill_gradient(uint32_t* buf, uint32_t stride_px, uint32_t w, uint32_t h,
                          uint32_t frame, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
  fill_gradient_kernel<<<grid, block, 0, stream>>>(buf, stride_px, w, h, frame);
}
