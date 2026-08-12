// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan::examples::v4l2_depth {

/// Convert a packed YUYV CUDA tensor to an opaque BGRA CUDA tensor.
class YuyvToBgraOp final : public holoscan::Operator<> {
 public:
  YuyvToBgraOp(std::int32_t width, std::int32_t height);

  void setup(holoscan::OperatorSpec& spec) override;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t width_;
  std::int32_t height_;
};

}  // namespace holoscan::examples::v4l2_depth
