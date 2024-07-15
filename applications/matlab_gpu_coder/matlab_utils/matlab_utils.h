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

/*
 * Utility code for using MATLAB GPU Coder generated CUDA with Holoscan SDK.
 *
 * NOTES
 * -Currently supports MATLAB function with one input tensor and one output tensor, and a variable
 * number of parameters.
 * -Currently supports input/output types: uint8/uint8 and float32/float32
 */

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

/**
 * @brief Equal to MATLAB's complex data type.
 */
typedef struct {
    float re, im;
} complex;

/**
 * @brief A 2D shape for use with hard_transpose_kernel_2d.
 */
struct Shape2 {
  int32_t ni;
  int32_t nj;
};

/**
 * @brief A 3D shape for use with hard_transpose_kernel_3d.
 */
struct Shape3 {
  int32_t ni;
  int32_t nj;
  int32_t nk;
};

/**
 * @brief Decides if the input shape to cuda_hard_transpose should be flipped.
 *
 * The flipping allows to move between row- and column-major based ordering.
 */
enum class Flip {
  Do,
  DoNot,
};

/**
 * @brief Changes elements in a tensor from row-major to column-major based ordering, and vice
 * versa.
 *
 * Reference:
 * https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
 */
template <typename T>
void cuda_hard_transpose(T* input, T* output, std::vector<int32_t> shape, cudaStream_t cuda_stream,
                         Flip flip);

/**
 * @brief Populates MATLAB complex data type complex from two arrays of floats
 */
void cuda_populate_complex(const float* rdata, const float* idata, void* output,
                           const int ndata, const cudaStream_t cuda_stream);

/**
 * @brief Creates a GXF tensor in device memory.
 *
 * OBS: Remember to delete when going out of scope!
 */
nvidia::gxf::Tensor* make_tensor(std::vector<int32_t>& shape, nvidia::gxf::PrimitiveType dtype_gxf,
                                 uint64_t bpp,
                                 nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator);
