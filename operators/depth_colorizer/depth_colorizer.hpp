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

/// Normalize a device depth map and resize it into a translucent BGRA jet overlay.
class DepthColorizerOp final : public holoscan::Operator<> {
 public:
  DepthColorizerOp(std::int32_t output_width,
                   std::int32_t output_height,
                   std::int32_t depth_width = 518,
                   std::int32_t depth_height = 518,
                   std::uint8_t alpha = 150U,
                   std::int32_t cuda_device = 0);
  ~DepthColorizerOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t output_width_;
  std::int32_t output_height_;
  std::int32_t depth_width_;
  std::int32_t depth_height_;
  std::uint8_t alpha_;
  std::int32_t cuda_device_;
  float* device_min_max_{};
};

}  // namespace holoscan::examples::v4l2_depth
