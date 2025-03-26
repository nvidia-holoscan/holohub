/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "convert_depth_to_screen_space.hpp"

namespace {

__global__ void convertDepthToScreenSpaceKernel(float* depth_buffer, int width, int height,
                                                float near_z, float far_z) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if ((px >= width) || (py >= height)) return;
  int pixel_index = py * width + px;

  float linear_depth = fmaxf(near_z, fminf(depth_buffer[pixel_index], far_z));
  depth_buffer[pixel_index] =
      ((linear_depth * far_z) - (near_z * far_z)) / (linear_depth * (far_z - near_z));
}

}  // namespace

void convertDepthToScreenSpace(cudaStream_t stream, float* depth_buffer, int width, int height,
                               float near_z, float far_z) {
  int tx = 8;
  int ty = 8;
  dim3 blocks(width / tx + 1, height / ty + 1);
  dim3 threads(tx, ty);
  convertDepthToScreenSpaceKernel<<<blocks, threads, 0, stream>>>(
      depth_buffer, width, height, near_z, far_z);
}
