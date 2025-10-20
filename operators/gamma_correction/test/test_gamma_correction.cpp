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

#include <gamma_correction/gamma_correction.hpp>

using namespace holoscan::ops;

/**
 * @brief Macro for safe CUDA Runtime API calls with automatic error handling
 *
 * This macro executes a CUDA Runtime statement and automatically checks for errors.
 * If the call fails, it throws a std::runtime_error with detailed information
 * including the statement, line number, file name, and CUDA error description.
 *
 * Usage:
 *   CUDA_CALL(cudaMalloc(&ptr, size));
 *
 * @param stmt The CUDA Runtime statement to execute
 * @param ... Additional parameters (unused, kept for compatibility)
 * @throws std::runtime_error if the CUDA call fails
 */
#define CUDA_CALL(stmt, ...)                                                               \
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
    spec.param(width_, "width", "Width of the output tensor.", "Width of the output tensor.", 16u);
    spec.param(component_count_,
               "component_count",
               "Component count of the output tensor.",
               "Component count of the output tensor.",
               1u);
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
    tensor->reshape<uint8_t>(nvidia::gxf::Shape({static_cast<int32_t>(width_.get()),
                                                 static_cast<int32_t>(component_count_.get())}),
                             nvidia::gxf::MemoryStorageType::kHost,
                             gxf_allocator.value());

    std::memcpy(tensor->pointer(), values_.get().data(), values_.get().size());

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::vector<uint8_t>> values_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> component_count_;
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
  explicit GammaCorrectionApp(const holoscan::ArgList& args = {}, uint32_t width = 1,
                              uint32_t component_count = 1)
      : args_(args) {
    width_ = width;
    component_count_ = component_count;

    input_values_.resize(width_ * component_count_);
    for (int i = 0; i < width_ * component_count_; i++) { input_values_[i] = 42 + i; }
  }

  void compose() override {
    auto op = make_operator<GammaCorrectionOp>(
        "gamma_correction_op", args_, make_condition<holoscan::CountCondition>(1));
    source_op_ = make_operator<SourceOp>("source_op",
                                         holoscan::Arg("width", width_),
                                         holoscan::Arg("component_count", component_count_),
                                         holoscan::Arg("values", input_values_));
    sink_op_ = make_operator<SinkOp>("sink_op");

    add_flow(source_op_, op);
    add_flow(op, sink_op_);
  }

  /**
   * Runs app and validates expected output in stdout.
   * Captures both stdout/stderr and checks for expected output and no errors.
   */
  void check_output(float gamma = 2.2f) {
    std::vector<uint8_t> expected_output(input_values_.size());

    const float range = (1 << (sizeof(uint8_t) * 8)) - 1;
    for (int i = 0; i < input_values_.size(); i++) {
      uint8_t input = input_values_[i];
      if (i % component_count_ < 3) {
        float fvalue = (float)input / range;
        fvalue = powf(fvalue, gamma);
        fvalue = fvalue * range + 0.5f;
        expected_output[i] = uint8_t(fvalue);
      } else {
        expected_output[i] = input;  // alpha channel is skipped
      }
    }

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
  uint32_t width_;
  uint32_t component_count_;
};

TEST(GammaCorrectionAppTest, DataTypeMissing) {
  auto application = holoscan::make_application<GammaCorrectionApp>();
  EXPECT_THROW(application->run(), std::runtime_error);
}

TEST(GammaCorrectionAppTest, Default) {
  auto application = holoscan::make_application<GammaCorrectionApp>(
      holoscan::ArgList({holoscan::Arg("data_type", "uint8_t")}));

  application->check_output();
}

TEST(GammaCorrectionAppTest, Gamma) {
  float gamma = 1.8f;
  auto application = holoscan::make_application<GammaCorrectionApp>(
      holoscan::ArgList({holoscan::Arg("data_type", "uint8_t"), holoscan::Arg("gamma", gamma)}));

  application->check_output(gamma);
}

TEST(GammaCorrectionAppTest, MultipleComponents) {
  const uint32_t width = 4;
  const int component_count = 4;
  auto application = holoscan::make_application<GammaCorrectionApp>(
      holoscan::ArgList({holoscan::Arg("data_type", "uint8_t"),
                         holoscan::Arg("component_count", component_count)}),
      width,
      component_count);

  application->check_output();
}
