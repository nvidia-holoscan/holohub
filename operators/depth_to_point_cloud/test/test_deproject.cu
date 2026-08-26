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

// Standalone golden-reference unit test for the deproject kernel. Compiles with
// nvcc alone (no Holoscan SDK) so kernel correctness can be verified on any GPU:
//   nvcc -O2 -arch=native -o test_deproject test/test_deproject.cu deproject.cu && ./test_deproject

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../deproject.hpp"

using namespace holoscan::ops;

static int g_failures = 0;
#define CHECK(cond, msg)                       \
  do {                                         \
    if (!(cond)) {                             \
      printf("FAIL: %s\n", (msg));             \
      ++g_failures;                            \
    }                                          \
  } while (0)

// Check that a CUDA runtime call returns cudaSuccess (allocations, copies, syncs).
#define CUDA_OK(call) CHECK((call) == cudaSuccess, #call " failed")

// CPU golden deproject for a single pixel.
static void golden(float z, int u, int v, CameraIntrinsics k, float dmin, float dmax, float invalid,
                   float& X, float& Y, float& Z) {
  if (z <= 0.f || z < dmin || z > dmax) {
    X = Y = Z = invalid;
    return;
  }
  X = (u - k.cx) * z / k.fx;
  Y = (v - k.cy) * z / k.fy;
  Z = z;
}

int main() {
  const int W = 64, H = 48, N = W * H;
  // cx,cy at image center: (W-1)/2, (H-1)/2
  const CameraIntrinsics k{50.f, 50.f, 31.5f, 23.5f};
  const float dmin = 0.1f, dmax = 10.f, invalid = NAN;

  // ---- Case 1: float32 constant plane Z=2.0, one hole -> NaN ----
  {
    std::vector<float> depth(N, 2.0f);
    depth[10 * W + 20] = 0.0f;  // hole
    float* d_depth = nullptr;
    float3* d_xyz = nullptr;
    CUDA_OK(cudaMalloc(&d_depth, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_xyz, N * sizeof(float3)));
    CUDA_OK(cudaMemcpy(d_depth, depth.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaError_t err = launch_deproject(d_depth, DepthDType::kFloat32, 1.0f, k, dmin, dmax, invalid,
                                       nullptr, 0, d_xyz, nullptr, W, H, 0);
    CHECK(err == cudaSuccess, "Case1 launch returned error");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float3> out(N);
    CUDA_OK(cudaMemcpy(out.data(), d_xyz, N * sizeof(float3), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int v = 0; v < H && ok; ++v) {
      for (int u = 0; u < W; ++u) {
        float gX, gY, gZ;
        golden(depth[v * W + u], u, v, k, dmin, dmax, invalid, gX, gY, gZ);
        float3 o = out[v * W + u];
        if (std::isnan(gX)) {
          if (!(std::isnan(o.x) && std::isnan(o.y) && std::isnan(o.z))) { ok = false; break; }
        } else if (fabsf(o.x - gX) > 1e-4f || fabsf(o.y - gY) > 1e-4f || fabsf(o.z - gZ) > 1e-4f) {
          ok = false;
          break;
        }
      }
    }
    CHECK(ok, "Case1 float32 plane deprojection mismatch / hole not NaN");
    cudaFree(d_depth);
    cudaFree(d_xyz);
  }

  // ---- Case 2: uint16 millimeters with depth_scale 0.001 ----
  {
    std::vector<uint16_t> depth(N, 2000);  // 2000 mm -> 2.0 m
    uint16_t* d_depth = nullptr;
    float3* d_xyz = nullptr;
    CUDA_OK(cudaMalloc(&d_depth, N * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_xyz, N * sizeof(float3)));
    CUDA_OK(cudaMemcpy(d_depth, depth.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));
    cudaError_t err = launch_deproject(d_depth, DepthDType::kUint16, 0.001f, k, dmin, dmax, invalid,
                                       nullptr, 0, d_xyz, nullptr, W, H, 0);
    CHECK(err == cudaSuccess, "Case2 launch returned error");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float3> out(N);
    CUDA_OK(cudaMemcpy(out.data(), d_xyz, N * sizeof(float3), cudaMemcpyDeviceToHost));
    const int u = 40, v = 30;
    float3 o = out[v * W + u];
    float gX = (u - k.cx) * 2.0f / k.fx, gY = (v - k.cy) * 2.0f / k.fy;
    CHECK(fabsf(o.z - 2.0f) < 1e-4f, "Case2 uint16 depth_scale Z wrong");
    CHECK(fabsf(o.x - gX) < 1e-4f && fabsf(o.y - gY) < 1e-4f, "Case2 uint16 XY wrong");
    cudaFree(d_depth);
    cudaFree(d_xyz);
  }

  // ---- Case 3: depth beyond depth_max -> invalid ----
  {
    std::vector<float> depth(N, 50.0f);  // > dmax
    float* d_depth = nullptr;
    float3* d_xyz = nullptr;
    CUDA_OK(cudaMalloc(&d_depth, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_xyz, N * sizeof(float3)));
    CUDA_OK(cudaMemcpy(d_depth, depth.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    cudaError_t err = launch_deproject(d_depth, DepthDType::kFloat32, 1.0f, k, dmin, dmax, invalid,
                                       nullptr, 0, d_xyz, nullptr, W, H, 0);
    CHECK(err == cudaSuccess, "Case3 launch returned error");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float3> out(N);
    CUDA_OK(cudaMemcpy(out.data(), d_xyz, N * sizeof(float3), cudaMemcpyDeviceToHost));
    CHECK(std::isnan(out[0].x), "Case3 depth beyond max not invalidated");
    cudaFree(d_depth);
    cudaFree(d_xyz);
  }

  // ---- Case 4: color passthrough (uchar3) ----
  {
    std::vector<float> depth(N, 2.0f);
    std::vector<uchar3> color(N);
    for (int i = 0; i < N; ++i) color[i] = make_uchar3(i % 256, (i * 2) % 256, (i * 3) % 256);
    float* d_depth = nullptr;
    float3* d_xyz = nullptr;
    uchar3* d_color = nullptr;
    uchar3* d_outcolor = nullptr;
    CUDA_OK(cudaMalloc(&d_depth, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_xyz, N * sizeof(float3)));
    CUDA_OK(cudaMalloc(&d_color, N * sizeof(uchar3)));
    CUDA_OK(cudaMalloc(&d_outcolor, N * sizeof(uchar3)));
    CUDA_OK(cudaMemcpy(d_depth, depth.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_color, color.data(), N * sizeof(uchar3), cudaMemcpyHostToDevice));
    cudaError_t err = launch_deproject(d_depth, DepthDType::kFloat32, 1.0f, k, dmin, dmax, invalid,
                                       d_color, 3, d_xyz, d_outcolor, W, H, 0);
    CHECK(err == cudaSuccess, "Case4 launch returned error");
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uchar3> outc(N);
    CUDA_OK(cudaMemcpy(outc.data(), d_outcolor, N * sizeof(uchar3), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < N; ++i) {
      if (outc[i].x != color[i].x || outc[i].y != color[i].y || outc[i].z != color[i].z) {
        ok = false;
        break;
      }
    }
    CHECK(ok, "Case4 color passthrough mismatch");
    cudaFree(d_depth);
    cudaFree(d_xyz);
    cudaFree(d_color);
    cudaFree(d_outcolor);
  }

  if (g_failures == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("%d FAILURE(S)\n", g_failures);
  return 1;
}
