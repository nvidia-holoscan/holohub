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

#pragma once

// CUDA kernel declarations for benchmarking
__global__ void simple_benchmark_kernel(float* data, const int size);

__global__ void background_load_kernel(float* data, const int size, const int intensity);

void async_run_simple_benchmark_kernel(
    float* data, int workload_size, int threads_per_block, cudaStream_t cuda_stream);

void async_run_background_load_kernel(
    float* data, int workload_size, int load_intensity, int threads_per_block,
    cudaStream_t cuda_stream);
