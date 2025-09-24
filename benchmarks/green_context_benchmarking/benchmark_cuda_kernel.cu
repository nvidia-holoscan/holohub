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

#include <cuda_runtime.h>
#include <cmath>
#include "benchmark_cuda_kernel.cu.hpp"

// CUDA kernels for benchmarking
__global__ void simple_benchmark_kernel(float* data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
    return;

  // Simple computation to create meaningful GPU work
  float value = (float)idx;
  value = value * 1.01f + 0.001f;
  value = sinf(value) + cosf(value);

  data[idx] = value;
}

__global__ void background_load_kernel(float* data, int size, int intensity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size || size == 0)
    return;

  float value = data[idx];

  // Create heavy computational load to stress GPU scheduler
  for (int i = 0; i < intensity; i++) {
    value = sinf(value) * cosf(value + idx);
    value += sqrtf(fabsf(value)) * 0.1f;
    value = fmaf(value, 1.01f, 0.001f);

    // Add memory access patterns
    if (i % 10 == 0) {
      value += data[(idx + i) % size] * 0.001f;
    }
  }
  data[idx] = value;
}

void async_run_simple_benchmark_kernel(
    float* data, int workload_size, int threads_per_block, cudaStream_t cuda_stream) {
  int blocks = (workload_size + threads_per_block - 1) / threads_per_block;
  simple_benchmark_kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
    data, workload_size);
}

void async_run_background_load_kernel(
    float* data, int workload_size, int load_intensity, int threads_per_block,
    cudaStream_t cuda_stream) {
  int blocks = (workload_size + threads_per_block - 1) / threads_per_block;
  background_load_kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
    data, workload_size, load_intensity);
}
