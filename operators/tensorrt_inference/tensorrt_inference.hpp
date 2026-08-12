// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

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
 * @brief Execute a static, single-input/single-output TensorRT network.
 *
 * The operator accepts either a serialized TensorRT engine or an ONNX model.
 * ONNX is built on the target GPU during `start()` and may be cached as a
 * serialized engine. Input and output tensors remain in plan-owned CUDA
 * device pools and are bound directly to TensorRT without staging copies.
 */
class TensorRtInferenceOp final : public holoscan::Operator<> {
 public:
  explicit TensorRtInferenceOp(std::string model_path, std::string engine_cache_path = {},
                               std::int32_t cuda_device = 0,
                               std::size_t max_output_bytes = 8U * 1024U * 1024U);
  ~TensorRtInferenceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  struct Impl;

  std::string model_path_;
  std::string engine_cache_path_;
  std::int32_t cuda_device_{};
  std::size_t max_output_bytes_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::examples::v4l2_depth
