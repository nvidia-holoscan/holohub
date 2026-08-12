// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace holoscan::examples::v4l2_depth {

/// Select one CUDA device for a scope and restore the caller's ambient device.
class CudaDeviceGuard final {
 public:
  explicit CudaDeviceGuard(std::int32_t requested_device) noexcept
      : requested_device_(requested_device) {
    if (requested_device_ < 0) {
      status_ = cudaErrorInvalidDevice;
      return;
    }
    status_ = cudaGetDevice(&previous_device_);
    if (status_ != cudaSuccess) {
      return;
    }
    if (previous_device_ != requested_device_) {
      status_ = cudaSetDevice(requested_device_);
      changed_ = status_ == cudaSuccess;
    }
    active_ = status_ == cudaSuccess;
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

  ~CudaDeviceGuard() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

  [[nodiscard]] std::string error_message(std::string_view operation) const {
    return std::string(operation) + " could not select CUDA device " +
           std::to_string(requested_device_) + ": " + cudaGetErrorString(status_);
  }

 private:
  std::int32_t requested_device_{};
  int previous_device_{};
  cudaError_t status_{cudaSuccess};
  bool changed_{};
  bool active_{};
};

}  // namespace holoscan::examples::v4l2_depth
