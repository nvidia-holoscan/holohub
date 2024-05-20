/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef MATLAB_UTILS_H
#define MATLAB_UTILS_H

#include "matlab_utils.h"

// Explicit instantiation
template void cuda_hard_transpose(float* input, float* output, std::vector<int32_t> shape,
                                  cudaStream_t cuda_stream, Flip flip);
template void cuda_hard_transpose(uint8_t* input, uint8_t* output, std::vector<int32_t> shape,
                                  cudaStream_t cuda_stream, Flip flip);

uint16_t ceil_div(uint16_t numerator, uint16_t denominator) {
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

template <typename T>
__global__ void hard_transpose_kernel_3d(T* input, T* output, Shape3 shape) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

  if ((i >= shape.ni) || (j >= shape.nj) || (k >= shape.nk)) { return; }

  const uint32_t r = k + shape.nk * (j + i * shape.nj);
  const uint32_t c = i + shape.ni * (j + k * shape.nj);

  output[c] = input[r];
}

template <typename T>
__global__ void hard_transpose_kernel_2d(T* input, T* output, Shape2 shape) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i >= shape.ni) || (j >= shape.nj)) { return; }

  const uint32_t r = j + i * shape.nj;
  const uint32_t c = i + j * shape.ni;

  output[c] = input[r];
}

__global__ void populate_complex_kernel(const float* rdata, const float* idata,
                                        complex* output, const int ndata) {
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < ndata; i += blockDim.x * gridDim.x) {
    float2 el = make_float2(rdata[i], idata[i]);
    float2 *tmp = (float2*)&output[i];
    *tmp = el;
  }
}

void cuda_populate_complex(const float* rdata, const float* idata, void* output,
                           const int ndata, const cudaStream_t cuda_stream) {
  int threadsPerBlock = 256;
  int numBlocks = std::ceil(ndata/(float)threadsPerBlock);
  populate_complex_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(
    rdata, idata, static_cast<complex*>(output), ndata);
}

template <typename T>
void cuda_hard_transpose(T* input, T* output, std::vector<int32_t> shape, cudaStream_t cuda_stream,
                         Flip flip) {
  if (flip == Flip::Do) std::reverse(shape.begin(), shape.end());

  // Set grid and block sizes
  dim3 threadsPerBlock;
  if (flip == Flip::Do) {
    threadsPerBlock = {1, 32, 32};
  } else {
    threadsPerBlock = {32, 32, 1};
  }
  dim3 numBlocks;
  numBlocks.x = std::ceil(shape[0]/(float)threadsPerBlock.x);
  numBlocks.y = std::ceil(shape[1]/(float)threadsPerBlock.y);
  if (shape.size() == 3) numBlocks.z = std::ceil(shape[2]/(float)threadsPerBlock.z);

  if (shape.size() == 3) {
    // 3D transpose
    hard_transpose_kernel_3d<T><<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(
        input, output, Shape3{shape[0], shape[1], shape[2]});
  } else if (shape.size() == 2) {
    // 2D transpose
    hard_transpose_kernel_2d<T>
        <<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(input, output, Shape2{shape[0], shape[1]});
  } else {
    throw std::runtime_error("Unsupported number of dimensions!");
  }
}

nvidia::gxf::Tensor* make_tensor(std::vector<int32_t>& shape, nvidia::gxf::PrimitiveType dtype_gxf,
                                 uint64_t bpp,
                                 nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  nvidia::gxf::Tensor* tensor = new nvidia::gxf::Tensor();
  tensor->reshapeCustom(nvidia::gxf::Shape(shape),
                        dtype_gxf,
                        bpp,
                        nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                        nvidia::gxf::MemoryStorageType::kDevice,
                        allocator);
  return tensor;
}

#endif
