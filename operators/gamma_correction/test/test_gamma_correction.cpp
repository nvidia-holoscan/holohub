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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <holoscan/holoscan.hpp>
#include <vector>

#include "gamma_correction.hpp"

#include "cuda_utils.hpp"

using namespace holoscan::ops;

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)
  SourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator for output buffers.",
               "Allocator for output buffers.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
    spec.param(values_, "values", "Values to be copied to the output tensor.");
    spec.output<nvidia::gxf::Entity>("output");
  }

  void initialize() override {
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_.default_value());

    // Call the base class initialize function
    Operator::initialize();
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto tensor =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("item").value();
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<Allocator>
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    tensor->reshape<uint8_t>(
        nvidia::gxf::Shape({16}), nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());

    std::memcpy(tensor->pointer(), values_.get().data(), values_.get().size());

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::vector<uint8_t>> values_;
};

class SinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto tensor = op_input.receive<std::shared_ptr<holoscan::Tensor>>("input").value();
    values_.resize(tensor->nbytes());
    CUDA_CALL(cudaMemcpy(values_.data(), tensor->data(), values_.size(), cudaMemcpyDefault));
  };

  std::vector<uint8_t> values_;
};

}  // namespace holoscan::ops

/**
 * Test application for GammaCorrectionOp integration testing.
 */
class GammaCorrectionApp : public holoscan::Application {
 public:
  /** Constructs app with shader source and optional args for SlangShaderOp */
  explicit GammaCorrectionApp(const holoscan::ArgList& args = {}) : args_(args) {
    input_values_.resize(16);
    for (int i = 0; i < 16; i++) { input_values_[i] = 42 + i; }
  }
  GammaCorrectionApp() = delete;

  void compose() override {
    auto op = make_operator<GammaCorrectionOp>(
        "gamma_correction_op", args_, make_condition<holoscan::CountCondition>(1));
    source_op_ = make_operator<SourceOp>(fmt::format("source_op"), holoscan::Arg("values", input_values_));
    sink_op_ = make_operator<SinkOp>(fmt::format("sink_op"));

    add_flow(source_op_, op);
    add_flow(op, sink_op_);
  }

  /**
   * Runs app and validates expected output in stdout.
   * Captures both stdout/stderr and checks for expected output and no errors.
   */
  void check_output(const std::vector<uint8_t>& expected_output) {
    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();

    EXPECT_NO_THROW(run());

    // Synchronize to ensure that the output is captured
    CUDA_CALL(cudaDeviceSynchronize());

    EXPECT_THAT(sink_op_->values_, testing::Eq(expected_output));

    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(stderr_output, testing::Not(testing::HasSubstr("error")));
  }

  std::shared_ptr<SourceOp> source_op_;
  std::shared_ptr<SinkOp> sink_op_;
  holoscan::ArgList args_;
  std::vector<uint8_t> input_values_;
};

// Test operator execution
TEST(SlangShaderAppTest, HelloWorld) {
  auto application = holoscan::make_application<GammaCorrectionApp>(
      holoscan::ArgList({holoscan::Arg("data_type", "uint8_t")}));

  float gamma = 2.2f;

  std::vector<uint8_t> expected_output(16);

  const float range = (1 << (sizeof(uint8_t) * 8)) - 1;
  for (int i = 0; i < 16; i++) {
    uint8_t input = application->input_values_[i];
    float fvalue = (float)input / range;
    fvalue = powf(fvalue, 1.f / gamma);
    fvalue = fvalue * range + 0.5f;
    expected_output[i] = uint8_t(fvalue);
  }

  application->check_output(expected_output);
}