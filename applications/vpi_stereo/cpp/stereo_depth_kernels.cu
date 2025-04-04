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

#include <stdio.h>
#include <iostream>
#include "stereo_depth_kernels.h"

__global__ void makeRectificationMapKernel(float* M, float* d, float* R, float* P, float* mapx,
                                           float* mapy, uint32_t width, uint32_t height) {
  uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u < width & v < height) {
    float x = (static_cast<float>(u) - P[2]) / P[0];
    float y = (static_cast<float>(v) - P[6]) / P[5];
    float z2 = R[2] * x + R[5] * y + R[8];
    float u2 = (R[0] * x + R[3] * y + R[6]) / z2;
    float v2 = (R[1] * x + R[4] * y + R[7]) / z2;
    float r2 = u2 * u2 + v2 * v2;
    float xy = u2 * v2;
    float rad = 1.0f + d[0] * r2 + d[1] * (r2 * r2) + d[4] * (r2 * r2 * r2);
    float u3 = rad * u2 + 2 * d[2] * xy + d[3] * (r2 + 2 * (u2 * u2));
    float v3 = rad * v2 + d[2] * (r2 + 2 * v2 * v2) + 2 * d[3] * xy;
    size_t tid = (u + v * width);
    mapx[tid] = M[0] * u3 + M[2];
    mapy[tid] = M[4] * v3 + M[5];
  }
}

void makeRectificationMap(float* M, float* d, float* R, float* P, float* mapx, float* mapy,
                          uint32_t width, uint32_t height, cudaStream_t stream) {
  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);
  makeRectificationMapKernel<<<launch_grid, block_dim, 0, stream>>>(
      M, d, R, P, mapx, mapy, width, height);
}

__global__ void heatmapF32Kernel(float* grayscale, uint8_t* rgb, float min_val, float inv_window,
                                 uint32_t width, uint32_t height) {
  uint32_t nx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ny = blockIdx.y * blockDim.y + threadIdx.y;
  if (nx < width & ny < height) {
    size_t tid_disp = (nx + ny * width);
    size_t tid_rgb = 3 * tid_disp;
    float v = (grayscale[tid_disp] - min_val) * inv_window;
    v = v > 0.0f ? v : 0.0f;
    v = v < 256.0f ? v : 0.0f;
    uint8_t r = static_cast<uint8_t>(v);
    rgb[tid_rgb] = r;
    rgb[tid_rgb + 1] = 0;
    rgb[tid_rgb + 2] = (r > 0) ? 255 - static_cast<uint8_t>(v) : 0;
  }
}

void heatmapF32(float* grayscale, uint8_t* rgb, float min_val, float max_val, uint32_t width,
                uint32_t height, cudaStream_t stream) {
  float inv_window = 255.0f / (max_val - min_val);
  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);
  heatmapF32Kernel<<<launch_grid, block_dim, 0, stream>>>(
      grayscale, rgb, min_val, inv_window, width, height);
}
