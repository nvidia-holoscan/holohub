// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/sample.hpp>

namespace holoscan::examples::v4l2_depth {

inline constexpr DLDataType kUInt8Dtype{kDLUInt, 8U, 1U};
inline constexpr DLDataType kFloat32Dtype{kDLFloat, 32U, 1U};

[[nodiscard]] constexpr bool same_dtype(DLDataType lhs, DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

[[nodiscard]] inline bool shape_equals(const holoscan::Tensor& tensor,
                                       std::span<const std::int64_t> expected) noexcept {
  const auto actual = tensor.shape_span();
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (actual[index] != expected[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool is_cuda_tensor(const holoscan::Tensor& tensor) noexcept {
  const DLDevice device = tensor.device();
  return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

[[nodiscard]] inline holoscan::Error invalid_tensor(std::string message) {
  return holoscan::Error{holoscan::ErrorCode::kInvalidArgument, std::move(message)};
}

[[nodiscard]] inline holoscan::EmitOptions forwarded_emit_options(
    const holoscan::SampleMetadata& metadata) {
  holoscan::EmitOptions options{.flags = metadata.flags};
  if (metadata.source_clock_id.valid()) {
    options.capture_time = metadata.capture_time();
  }
  if (metadata.frame_id != 0U) {
    options.frame_id = metadata.frame_id;
  }
  if (metadata.trace_id != 0U) {
    options.trace = holoscan::TraceContext::root(metadata.trace_id);
  }
  return options;
}

}  // namespace holoscan::examples::v4l2_depth
