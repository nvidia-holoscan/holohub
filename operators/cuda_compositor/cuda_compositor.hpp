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

/// Latch overlay updates and composite the latest overlay over every base frame.
class CudaCompositorOp final : public holoscan::Operator<> {
 public:
  CudaCompositorOp(std::int32_t width,
                   std::int32_t height,
                   std::int32_t cuda_device = 0);
  ~CudaCompositorOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> base;
  holoscan::Input<holoscan::Tensor> overlay;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t width_;
  std::int32_t height_;
  std::int32_t cuda_device_;
  std::uint8_t* overlay_latch_{};
};

}  // namespace holoscan::examples::v4l2_depth
