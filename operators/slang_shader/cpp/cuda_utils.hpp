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

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <holoscan/logger/logger.hpp>
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
 *   CUDA_RT_CALL(cudaMalloc(&ptr, size));
 *
 * @param stmt The CUDA Runtime statement to execute
 * @param ... Additional parameters (unused, kept for compatibility)
 * @throws std::runtime_error if the CUDA call fails
 */
#define CUDA_RT_CALL(stmt, ...)                                                            \
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
 * @brief Macro for safe CUDA driver API calls with automatic error handling
 *
 * This macro executes a CUDA driver statement and automatically checks for errors.
 * If the call fails, it throws a std::runtime_error with detailed information
 * including the statement, line number, file name, and CUDA error description.
 *
 * Usage:
 *   CUDA_CALL(cuMemAlloc(&ptr, size));
 *
 * @param stmt The CUDA driver statement to execute
 * @param ... Additional parameters (unused, kept for compatibility)
 * @throws std::runtime_error if the CUDA call fails
 */
#define CUDA_CALL(stmt, ...)                                                              \
  ({                                                                                      \
    CUresult _holoscan_cuda_err = stmt;                                                   \
    if (CUDA_SUCCESS != _holoscan_cuda_err) {                                             \
      const char* error_string = "";                                                      \
      cuGetErrorString(_holoscan_cuda_err, &error_string);                                \
      throw std::runtime_error(                                                           \
          fmt::format("CUDA driver call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                              \
                      __LINE__,                                                           \
                      __FILE__,                                                           \
                      error_string,                                                       \
                      static_cast<int>(_holoscan_cuda_err)));                             \
    }                                                                                     \
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
template <typename T, CUresult func(T)>
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
 * Helper class for using handles with std::unique_ptr which requires that a custom
 * handle type satisfies NullablePointer
 * https://en.cppreference.com/w/cpp/named_req/NullablePointer.
 *
 * @tparam T type to hold
 */
template <typename T>
class Nullable {
 public:
  Nullable(T value = 0) : value_(value) {}
  Nullable(std::nullptr_t) : value_(0) {}
  operator T() const { return value_; }
  explicit operator bool() { return value_ != 0; }

  friend bool operator==(Nullable l, Nullable r) { return l.value_ == r.value_; }
  friend bool operator!=(Nullable l, Nullable r) { return !(l == r); }

  /**
   * Deleter, call the function when the object is deleted.
   *
   * @tparam F function to call
   */
  template <typename RESULT, RESULT func(T)>
  struct Deleter {
    typedef Nullable<T> pointer;
    void operator()(T value) const { func(value); }
  };

 private:
  T value_;
};

/**
 * @brief Type alias for a unique_ptr that manages CUDA library handles
 *
 * This type provides automatic cleanup of CUDA library handles using
 * cudaLibraryUnload when the unique_ptr goes out of scope.
 */
using UniqueCuLibrary = std::unique_ptr<CUlibrary, Deleter<CUlibrary, &cuLibraryUnload>>;

/**
 * @brief Type alias for a unique_ptr that manages CUDA device primary context
 *
 * This type provides automatic cleanup of CUDA device primary context using
 * cuDevicePrimaryCtxRelease when the unique_ptr goes out of scope.
 */
using UniqueDevicePrimaryContext =
    std::unique_ptr<Nullable<CUdevice>,
                    Nullable<CUdevice>::Deleter<CUresult, &cuDevicePrimaryCtxRelease>>;

class ScopedPushCuContext {
 public:
  /**
   * @brief Construct a new scoped cuda context object
   *
   * @param cuda_context context to push
   */
  explicit ScopedPushCuContext(CUcontext cuda_context) : cuda_context_(cuda_context) {
    // might be called from a different thread than the thread
    // which constructed CudaPrimaryContext, therefore call cuInit()
    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuCtxPushCurrent(cuda_context_));
  }
  ScopedPushCuContext() = delete;

  ~ScopedPushCuContext() {
    try {
      CUcontext popped_context;
      CUDA_CALL(cuCtxPopCurrent(&popped_context));
      if (popped_context != cuda_context_) {
        HOLOSCAN_LOG_ERROR("Cuda: Unexpected context popped");
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("ScopedPushCuContext destructor failed with {}", e.what());
    }
  }

 private:
  const CUcontext cuda_context_;
};

}  // namespace holoscan::ops

#endif /* CUDA_UTILS_HPP */
