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

#ifndef HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPROJECT_HPP
#define HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPROJECT_HPP

#include <cuda_runtime.h>

#include <cstdint>

// This header is intentionally free of any Holoscan dependency so the deproject
// kernel can be unit-tested in isolation with nvcc (see test/test_deproject.cu).

namespace holoscan::ops {

/// Supported element types of the input depth image.
enum class DepthDType : int { kUint16 = 0, kFloat32 = 1 };

/// Pinhole camera intrinsics in pixels.
struct CameraIntrinsics {
  float fx;
  float fy;
  float cx;
  float cy;
};

/**
 * @brief Deproject an organized depth image into an organized HxWx3 point cloud.
 *
 * One CUDA thread per pixel (grid-stride). For each pixel (u, v) with metric depth
 * Z = raw_depth * depth_scale, the output point (optical frame: x-right, y-down,
 * z-forward) is:
 *     X = (u - cx) * Z / fx
 *     Y = (v - cy) * Z / fy
 *     Z = Z
 * Pixels with raw depth == 0 or metric depth outside [depth_min, depth_max] are
 * written as (invalid_value, invalid_value, invalid_value).
 *
 * @param depth          device pointer to the HxW depth image (element type = dtype)
 * @param dtype          depth element type (uint16 or float32)
 * @param depth_scale    multiply raw depth by this to get meters (e.g. 0.001 for uint16 mm)
 * @param intr           pinhole intrinsics in pixels
 * @param depth_min      minimum valid metric depth (meters), inclusive
 * @param depth_max      maximum valid metric depth (meters), inclusive
 * @param invalid_value  value written to X/Y/Z for invalid pixels (e.g. NaN or 0)
 * @param color          optional device pointer to HxW color (uchar3/uchar4); nullptr to disable
 * @param color_channels 3 or 4; ignored when color == nullptr
 * @param out_xyz        device pointer to HxW float3 output (organized, AoS)
 * @param out_color      optional device pointer to HxW uchar3 output; nullptr to disable
 * @param width          image width in pixels
 * @param height         image height in pixels
 * @param stream         CUDA stream to launch on
 * @return cudaPeekAtLastError() result after the launch
 */
cudaError_t launch_deproject(const void* depth, DepthDType dtype, float depth_scale,
                             CameraIntrinsics intr, float depth_min, float depth_max,
                             float invalid_value, const void* color, int color_channels,
                             float3* out_xyz, uchar3* out_color, int width, int height,
                             cudaStream_t stream);

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_DEPTH_TO_POINT_CLOUD_DEPROJECT_HPP */
