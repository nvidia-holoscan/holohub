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
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::examples::v4l2_depth {

/**
 * @brief Drop all but every Nth CUDA tensor without copying retained frames.
 *
 * Every input is received, so the upstream channel continues to drain. Retained
 * samples are forwarded as complete `Sample<Tensor>` values, preserving
 * capture time, clock provenance, frame ID, trace, and semantic flags.
 */
class FrameSkipperOp final : public holoscan::Operator<> {
 public:
  explicit FrameSkipperOp(std::uint32_t keep_one_in_n);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::uint32_t keep_one_in_n_{};
  std::uint64_t received_{};
};

}  // namespace holoscan::examples::v4l2_depth
