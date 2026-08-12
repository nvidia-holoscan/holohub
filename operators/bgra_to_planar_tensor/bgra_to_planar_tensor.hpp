// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan::examples::v4l2_depth {

/// Resize a BGRA tensor and produce a normalized planar RGB model input tensor.
class BgraToPlanarTensorOp final : public holoscan::Operator<> {
 public:
  BgraToPlanarTensorOp(
      std::int32_t source_width,
      std::int32_t source_height,
      std::int32_t network_width = 518,
      std::int32_t network_height = 518,
      std::array<float, 3> mean = {0.485F, 0.456F, 0.406F},
      std::array<float, 3> standard_deviation = {0.229F, 0.224F, 0.225F});

  void setup(holoscan::OperatorSpec& spec) override;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t source_width_;
  std::int32_t source_height_;
  std::int32_t network_width_;
  std::int32_t network_height_;
  std::array<float, 3> mean_;
  std::array<float, 3> standard_deviation_;
};

}  // namespace holoscan::examples::v4l2_depth
