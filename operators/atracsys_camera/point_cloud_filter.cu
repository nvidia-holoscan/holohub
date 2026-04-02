/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "point_cloud_filter.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace atracsys::ops {

__global__ void point_cloud_filter_kernel(const int16_t* __restrict__ d_disp,
                                          int width,
                                          int height,
                                          size_t disp_step_bytes,
                                          float q00, float q01, float q02, float q03,
                                          float q10, float q11, float q12, float q13,
                                          float q20, float q21, float q22, float q23,
                                          float q30, float q31, float q32, float q33,
                                          float min_z, float max_z,
                                          float max_x, float max_y,
                                          float* __restrict__ d_out_points) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

  int idx = v * width + u;

  auto invalidate = [&] {
    d_out_points[idx * 3 + 0] = nanf("");
    d_out_points[idx * 3 + 1] = nanf("");
    d_out_points[idx * 3 + 2] = nanf("");
  };

  const int16_t* row_ptr = (const int16_t*)((const char*)d_disp + v * disp_step_bytes);
  int16_t raw_d = row_ptr[u];

  float d = static_cast<float>(raw_d) / 16.0f;
  if (d <= 0.0f) { invalidate(); return; }

  float W = q30 * u + q31 * v + q32 * d + q33;
  if (W == 0.0f) { invalidate(); return; }

  float X = (q00 * u + q01 * v + q02 * d + q03) / W;
  float Y = (q10 * u + q11 * v + q12 * d + q13) / W;
  float Z = (q20 * u + q21 * v + q22 * d + q23) / W;

  if (!isfinite(X) || !isfinite(Y) || !isfinite(Z)) { invalidate(); return; }
  if (Z <= min_z || Z >= max_z || fabsf(X) >= max_x || fabsf(Y) >= max_y) { invalidate(); return; }

  d_out_points[idx * 3 + 0] = X;
  d_out_points[idx * 3 + 1] = Y;
  d_out_points[idx * 3 + 2] = Z;
}

void launch_point_cloud_filter(const int16_t* d_disp,
                               int width,
                               int height,
                               size_t disp_step_bytes,
                               const float* h_Q,
                               float min_z,
                               float max_z,
                               float max_x,
                               float max_y,
                               float* d_out_points,
                               void* stream) {
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

  dim3 blocks((width + 31) / 32, (height + 31) / 32);
  dim3 threads(32, 32);

  point_cloud_filter_kernel<<<blocks, threads, 0, cu_stream>>>(
      d_disp, width, height, disp_step_bytes,
      h_Q[0], h_Q[1], h_Q[2], h_Q[3],
      h_Q[4], h_Q[5], h_Q[6], h_Q[7],
      h_Q[8], h_Q[9], h_Q[10], h_Q[11],
      h_Q[12], h_Q[13], h_Q[14], h_Q[15],
      min_z, max_z, max_x, max_y,
      d_out_points);
}

}  // namespace atracsys::ops
