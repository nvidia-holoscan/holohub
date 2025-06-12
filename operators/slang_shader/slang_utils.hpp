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

#ifndef SLANG_UTILS_HPP
#define SLANG_UTILS_HPP

#include <cuda_runtime.h>
#include <slang-com-helper.h>
#include <holoscan/logger/logger.hpp>
#include <memory>

namespace holoscan::ops {

/**
 * @brief Converts a Slang result code to a human-readable error string
 *
 * @param result The Slang result code to convert
 * @return const char* A string describing the error, or "Unknown error" if the result is not
 * recognized
 */
const char* get_error_string(Slang::Result result);

/**
 * @brief Macro for safe Slang API calls with automatic error handling
 *
 * This macro executes a Slang statement and automatically checks for errors.
 * If the call fails, it throws a std::runtime_error with detailed information
 * including the statement, line number, file name, and error description.
 *
 * Usage:
 *   SLANG_CALL(slangCreateSession(&session));
 *
 * @param stmt The Slang statement to execute
 * @return Slang::Result The result of the statement execution
 * @throws std::runtime_error if the Slang call fails
 */
#define SLANG_CALL(stmt)                                                            \
  ({                                                                                \
    Slang::Result _slang_result = stmt;                                             \
    if (SLANG_FAILED(_slang_result)) {                                              \
      throw std::runtime_error(                                                     \
          fmt::format("Slang call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                        \
                      __LINE__,                                                     \
                      __FILE__,                                                     \
                      get_error_string(_slang_result),                              \
                      static_cast<int>(_slang_result)));                            \
    }                                                                               \
    _slang_result;                                                                  \
  })

/**
 * @brief Macro for safe Slang API calls with diagnostics support
 *
 * This macro is similar to SLANG_CALL but includes diagnostic information
 * in the error message when available. It's useful for compilation errors
 * where additional diagnostic details are provided by Slang.
 *
 * Usage:
 *   SLANG_CALL_WITH_DIAGNOSTICS(slangCompileRequestCompile(request, diagnostics_blob));
 *
 * @param stmt The Slang statement to execute
 * @return Slang::Result The result of the statement execution
 * @throws std::runtime_error if the Slang call fails, including diagnostic information
 */
#define SLANG_CALL_WITH_DIAGNOSTICS(stmt)                                               \
  ({                                                                                    \
    Slang::Result _slang_result = stmt;                                                 \
    if (SLANG_FAILED(_slang_result)) {                                                  \
      throw std::runtime_error(fmt::format(                                             \
          "Slang call {} in line {} of file {} failed with '{}' ({}), diagnostics: {}", \
          #stmt,                                                                        \
          __LINE__,                                                                     \
          __FILE__,                                                                     \
          get_error_string(_slang_result),                                              \
          static_cast<int>(_slang_result),                                              \
          (const char*)diagnostics_blob->getBufferPointer()));                          \
    }                                                                                   \
    _slang_result;                                                                      \
  })

/**
 * @brief Macro to log Slang compilation diagnostics if available
 *
 * This macro checks if diagnostic information is available and logs it
 * at INFO level. It's typically used after compilation operations to
 * provide additional debugging information.
 *
 * Usage:
 *   SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
 *
 * @param diagnostics_blob Pointer to the diagnostics blob containing compilation information
 */
#define SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob)                        \
  if (diagnostics_blob) {                                                 \
    HOLOSCAN_LOG_INFO("Slang compilation diagnostics: {}",                \
                      (const char*)diagnostics_blob->getBufferPointer()); \
  }

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

}  // namespace holoscan::ops

#endif /* SLANG_UTILS_HPP */
