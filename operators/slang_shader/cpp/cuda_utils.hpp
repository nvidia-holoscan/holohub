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

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <memory>

namespace holoscan::ops {

/**
 * @brief Macro for safe CUDA Runtime API calls with automatic error handling
 *
 * This macro executes a CUDA Runtime statement and automatically checks for errors.
 * If the call fails, it throws a std::runtime_error with detailed information
 * including the statement, line number, file name, and CUDA error description.
 *
 * Usage:
 *   CUDA_CALL(cudaMalloc(&ptr, size));
 *
 * @param stmt The CUDA Runtime statement to execute
 * @param ... Additional parameters (unused, kept for compatibility)
 * @throws std::runtime_error if the CUDA call fails
 */
#define CUDA_CALL(stmt, ...)                                                               \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      throw std::runtime_error(                                                            \
          fmt::format("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                               \
                      __LINE__,                                                            \
                      __FILE__,                                                            \
                      cudaGetErrorString(_holoscan_cuda_err),                              \
                      static_cast<int>(_holoscan_cuda_err)));                              \
    }                                                                                      \
  })

/**
 * @brief Custom deleter for CUDA objects managed by unique_ptr
 *
 * This template provides a custom deleter for CUDA objects that need
 * to be properly cleaned up when managed by std::unique_ptr. It calls
 * the specified cleanup function when the unique_ptr goes out of scope.
 *
 * @tparam T The CUDA object type
 * @tparam func The cleanup function to call (e.g., cudaLibraryUnload)
 */
template <typename T, cudaError_t func(T)>
struct Deleter {
  typedef T pointer;
  /**
   * @brief Operator to call the cleanup function
   *
   * @param value The CUDA object to clean up
   */
  void operator()(T value) const { func(value); }
};

/**
 * @brief Type alias for a unique_ptr that manages CUDA library handles
 *
 * This type provides automatic cleanup of CUDA library handles using
 * cudaLibraryUnload when the unique_ptr goes out of scope.
 */
using UniqueCudaLibrary =
    std::unique_ptr<cudaLibrary_t, Deleter<cudaLibrary_t, &cudaLibraryUnload>>;

/**
 * @brief Type alias for a unique_ptr that manages CUDA host memory
 *
 * This type provides automatic cleanup of CUDA host memory using cudaFreeHost
 * when the unique_ptr goes out of scope.
 */
using UniqueCudaHostMemory = std::unique_ptr<void, Deleter<void*, &cudaFreeHost>>;

}  // namespace holoscan::ops

#endif /* CUDA_UTILS_HPP */
