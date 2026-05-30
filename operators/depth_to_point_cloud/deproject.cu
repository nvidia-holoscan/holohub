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

#include "deproject.hpp"

namespace holoscan::ops {

// Pattern 1 (map / elementwise): one thread per pixel, grid-stride loop.
template <typename T>
__global__ void __launch_bounds__(256) deproject_kernel(
    const T* __restrict__ depth, float depth_scale, CameraIntrinsics k, float depth_min,
    float depth_max, float invalid_value, const uchar3* __restrict__ color3,
    const uchar4* __restrict__ color4, float3* __restrict__ out_xyz,
    uchar3* __restrict__ out_color, int width, int height) {
  const int n = width * height;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const int u = i % width;
    const int v = i / width;

    const T raw = depth[i];
    const float z = static_cast<float>(raw) * depth_scale;
    // Use a positive-depth check (rather than `raw != 0`) so that negative and NaN depths
    // are uniformly rejected for both the uint16 and float32 paths.
    const bool valid = (z > 0.0f) && isfinite(z) && (z >= depth_min) && (z <= depth_max);

    out_xyz[i] = valid ? make_float3((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z)
                       : make_float3(invalid_value, invalid_value, invalid_value);

    if (out_color != nullptr) {
      if (color4 != nullptr) {
        const uchar4 c = color4[i];
        out_color[i] = make_uchar3(c.x, c.y, c.z);
      } else if (color3 != nullptr) {
        out_color[i] = color3[i];
      }
    }
  }
}

cudaError_t launch_deproject(const void* depth, DepthDType dtype, float depth_scale,
                             CameraIntrinsics intr, float depth_min, float depth_max,
                             float invalid_value, const void* color, int color_channels,
                             float3* out_xyz, uchar3* out_color, int width, int height,
                             cudaStream_t stream) {
  const int n = width * height;
  if (n <= 0) { return cudaSuccess; }

  constexpr int kBlock = 256;
  int grid = (n + kBlock - 1) / kBlock;
  // Grid-stride loop handles any n; cap grid to keep launch configuration sane.
  grid = grid < 65535 ? grid : 65535;

  const uchar3* color3 = nullptr;
  const uchar4* color4 = nullptr;
  if (color != nullptr && out_color != nullptr) {
    if (color_channels == 4) {
      color4 = static_cast<const uchar4*>(color);
    } else if (color_channels == 3) {
      color3 = static_cast<const uchar3*>(color);
    } else {
      return cudaErrorInvalidValue;  // only 3- or 4-channel uint8 color is supported
    }
  }

  switch (dtype) {
    case DepthDType::kUint16:
      deproject_kernel<uint16_t><<<grid, kBlock, 0, stream>>>(
          static_cast<const uint16_t*>(depth), depth_scale, intr, depth_min, depth_max,
          invalid_value, color3, color4, out_xyz, out_color, width, height);
      break;
    case DepthDType::kFloat32:
      deproject_kernel<float><<<grid, kBlock, 0, stream>>>(
          static_cast<const float*>(depth), depth_scale, intr, depth_min, depth_max, invalid_value,
          color3, color4, out_xyz, out_color, width, height);
      break;
    default:
      return cudaErrorInvalidValue;  // unknown depth dtype -> no kernel launched
  }

  return cudaPeekAtLastError();
}

}  // namespace holoscan::ops
