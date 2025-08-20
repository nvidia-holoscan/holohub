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

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <holoscan/holoscan.hpp>

#include <slang_shader/slang_shader.hpp>

#include "../cuda_utils.hpp"

using namespace holoscan::ops;

class SlangShaderOpTest : public ::testing::Test {
 protected:
  void SetUp() override { fragment_ = std::make_shared<holoscan::Fragment>(); }

  void TearDown() override { fragment_.reset(); }

  std::shared_ptr<holoscan::Fragment> fragment_;
};

static const char g_simple_shader[] = R"(
  import holoscan;

  [holoscan::input("input_buffer")]
  StructuredBuffer<int> input_buffer;

  [holoscan::output("output_buffer")]
  [holoscan::alloc::size_of("input_buffer")]
  RWStructuredBuffer<int> output_buffer;

  [shader("compute")]
  void compute(uint3 gid : SV_DispatchThreadID) {
    output_buffer[gid.x] = input_buffer[gid.x];
  }
)";

// Test basic operator construction
TEST_F(SlangShaderOpTest, Construction) {
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<SlangShaderOp>(
        "slang_shader_op", holoscan::Arg("shader_source", g_simple_shader));
    EXPECT_NE(op, nullptr);
  });
}

// Test operator setup with empty shader source (should throw)
TEST_F(SlangShaderOpTest, SetupWithEmptyShaderSource) {
  EXPECT_THROW(fragment_->make_operator<SlangShaderOp>("slang_shader_op",
                                                       holoscan::Arg("shader_source", "")),
               std::runtime_error);
}

// Test operator setup with invalid shader file (should throw)
TEST_F(SlangShaderOpTest, SetupWithInvalidShaderFile) {
  EXPECT_THROW(fragment_->make_operator<SlangShaderOp>(
                   "slang_shader_op", holoscan::Arg("shader_source_file", "some_file.slang")),
               std::runtime_error);
}

// Test operator setup with both shader_source and shader_source_file (should throw)
TEST_F(SlangShaderOpTest, SetupWithBothShaderSources) {
  EXPECT_THROW(fragment_->make_operator<SlangShaderOp>(
                   "slang_shader_op",
                   holoscan::Arg("shader_source", "some shader code"),
                   holoscan::Arg("shader_source_file", "some_file.slang")),
               std::runtime_error);
}

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)
  SourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(shape_, "shape", "Shape of the output buffer.", "Shape of the output buffer.");
    spec.param(allocator_,
               "allocator",
               "Allocator for output buffers.",
               "Allocator for output buffers.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
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
    tensor->reshape<int>(nvidia::gxf::Shape(shape_.get()),
                         nvidia::gxf::MemoryStorageType::kHost,
                         gxf_allocator.value());

    std::vector<int> value(tensor->element_count());
    for (int i = 0; i < value.size(); i++) { value[i] = 42 + i; }
    std::memcpy(tensor->pointer(), value.data(), value.size() * sizeof(int));

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::vector<int32_t>> shape_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

class VideoBufferSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoBufferSourceOp)
  VideoBufferSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator for output buffers.",
               "Allocator for output buffers.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
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
    auto video_buffer =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>("item").value();
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<Allocator>
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    video_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        2,
        3,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
        nvidia::gxf::MemoryStorageType::kHost,
        gxf_allocator.value(),
        false /* stride_align */);

    uint8_t value[2 * 3 * 4];
    for (int i = 0; i < 2 * 3 * 4; i++) { value[i] = 42 + i; }
    std::memcpy(video_buffer->pointer(), value, sizeof(value));

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

class SinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("input");
    spec.param(expected_shape_,
               "expected_shape",
               "Expected shape of the input buffer.",
               "Expected shape of the input buffer.");
    spec.param(print_output_,
               "print_output",
               "Print the output buffer.",
               "Print the output buffer.",
               false);
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto tensor = op_input.receive<std::shared_ptr<holoscan::Tensor>>("input").value();

    std::vector<int32_t> current_shape;
    std::vector<int64_t> tensor_shape = tensor->shape();
    std::transform(tensor_shape.begin(),
                   tensor_shape.end(),
                   std::back_inserter(current_shape),
                   [](int64_t value) { return static_cast<int32_t>(value); });
    EXPECT_EQ(current_shape, expected_shape_.get());

    if (print_output_.get()) {
      std::vector<int> values(tensor->size());
      CUDA_RT_CALL(cudaMemcpy(
          values.data(), tensor->data(), tensor->size() * sizeof(int), cudaMemcpyDefault));

      std::cout << fmt::format("{}", fmt::join(values, ", ")) << std::endl;
    }
  };

 private:
  Parameter<std::vector<int32_t>> expected_shape_;
  Parameter<bool> print_output_;
};

}  // namespace holoscan::ops

/**
 * Test application for SlangShaderOp integration testing.
 * Supports flexible pipeline composition with optional source/sink operators.
 */
class SlangShaderApp : public holoscan::Application {
 public:
  /** Constructs app with shader source and optional args for SlangShaderOp */
  explicit SlangShaderApp(const std::string& shader_source, const holoscan::ArgList& args = {})
      : args_(args) {
    args_.add(holoscan::Arg("shader_source", shader_source));
  }
  SlangShaderApp() = delete;

  /** Sets the shape of the input tensor */
  void set_input_shape(const std::vector<int32_t>& shape) { input_shape_ = shape; }

  /** Enables source operator that generates test data (tensor with value 42) */
  void add_source_op(const std::string& input = "input") { source_ops_.push_back(input); }

  /** Enables source operator that generates test data (video buffer with value 42) */
  void add_video_buffer_source_op() { add_video_buffer_source_op_ = true; }

  /** Sets the shape of the output tensor */
  void set_output_shape(const std::vector<int32_t>& shape) { output_shape_ = shape; }

  /** Enables prints from output operator */
  void print_output() { print_output_ = true; }

  /** Enables sink operator that validates SlangShaderOp output */
  void add_sink_op(const std::string& input = "input") { sink_ops_.push_back(input); }

  /**
   * Composes pipeline based on enabled flags:
   * - SourceOp -> SlangShaderOp (if add_source_op_)
   * - SlangShaderOp -> SinkOp (if add_sink_op_)
   * - SourceOp -> SlangShaderOp -> SinkOp (if both)
   * - SlangShaderOp only (if neither)
   */
  void compose() override {
    auto op = make_operator<SlangShaderOp>(
        "slang_shader_op", args_, make_condition<holoscan::CountCondition>(1));
    if (source_ops_.size() == 1) {
      auto source_op = make_operator<SourceOp>("source_op", holoscan::Arg("shape", input_shape_));
      add_flow(source_op, op);
    } else {
      for (const auto& input : source_ops_) {
        auto source_op = make_operator<SourceOp>(fmt::format("source_op_{}", input),
                                                 holoscan::Arg("shape", input_shape_));
        add_flow(source_op, op, {{"output", input}});
      }
    }
    if (add_video_buffer_source_op_) {
      auto source_op = make_operator<VideoBufferSourceOp>("video_buffer_source_op");
      add_flow(source_op, op);
    }
    if (sink_ops_.size() == 1) {
      auto sink_op = make_operator<SinkOp>("sink_op",
                                           holoscan::Arg("expected_shape", output_shape_),
                                           holoscan::Arg("print_output", print_output_));
      add_flow(op, sink_op);
    } else {
      for (const auto& output : sink_ops_) {
        auto sink_op = make_operator<SinkOp>(fmt::format("sink_op_{}", output),
                                             holoscan::Arg("expected_shape", output_shape_),
                                             holoscan::Arg("print_output", print_output_));
        add_flow(op, sink_op, {{output, "input"}});
      }
    }
    if (source_ops_.empty() && sink_ops_.empty()) {
      add_operator(op);
    }
  }

  /**
   * Runs app and validates expected output in stdout.
   * Captures both stdout/stderr and checks for expected output and no errors.
   */
  void check_output(std::string expected_output) {
    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();
    testing::internal::CaptureStdout();

    EXPECT_NO_THROW(run());

    // Synchronize to ensure that the output is captured
    CUDA_RT_CALL(cudaDeviceSynchronize());

    std::string stdout_output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(stdout_output, testing::Eq(expected_output));

    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_THAT(stderr_output, testing::Not(testing::HasSubstr("error")));
  }

  holoscan::ArgList args_;               ///< Arguments passed to the SlangShaderOp
  std::vector<std::string> source_ops_;  ///< Flag to enable source operator in pipeline
  bool add_video_buffer_source_op_ =
      false;                           ///< Flag to enable video buffer source operator in pipeline
  std::vector<std::string> sink_ops_;  ///< Flag to enable sink operator in pipeline
  std::vector<int32_t> input_shape_ = {1};   ///< Shape of the input tensor
  std::vector<int32_t> output_shape_ = {1};  ///< Shape of the output tensor
  bool print_output_ = false;                ///< Flag to enable prints from output operator
};

// Test operator execution
TEST(SlangShaderAppTest, HelloWorld) {
  const std::string shader_source = R"(
    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("Hello, world!");
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->check_output("Hello, world!");
}

TEST(SlangShaderAppTest, Parameter) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::parameter("parameter")]
    float parameter;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%f", parameter);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(
      shader_source, holoscan::ArgList({holoscan::Arg("parameter", 1.0f)}));

  application->check_output("1.000000");
}

TEST(SlangShaderAppTest, ParameterWithDefault) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::parameter("parameter=12")]
    int parameter;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d", parameter);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);

  application->check_output("12");
}

TEST(SlangShaderAppTest, InputTensor) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<int> input_buffer;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d", input_buffer[gid.x]);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op();

  application->check_output("42");
}

TEST(SlangShaderAppTest, InputTensorMultiDimensional) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<int> input_buffer;

    [holoscan::size_of("input_buffer")]
    uint3 size;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d %d %d\n", size.x, size.y, size.z);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_input_shape({4, 3, 2, 1});
  application->add_source_op();

  // shape is DHWC (depth, height, width, channels) which is 4, 3, 2, 1
  // size is W, H, D (width, height, depth) which is 2, 3, 4
  application->check_output("2 3 4\n");
}

TEST(SlangShaderAppTest, InputMultipleTensors) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer_1")]
    StructuredBuffer<int> input_buffer_1;

    [holoscan::input("input_buffer_2")]
    StructuredBuffer<int> input_buffer_2;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d %d", input_buffer_1[gid.x], input_buffer_2[gid.x]);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op("input_buffer_1");
  application->add_source_op("input_buffer_2");

  application->check_output("42 42");
}

TEST(SlangShaderAppTest, InputTensorMap) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer:item")]
    StructuredBuffer<int> input_buffer;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d", input_buffer[gid.x]);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op();

  application->check_output("42");
}

TEST(SlangShaderAppTest, InputVideoBuffer) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer:item")]
    StructuredBuffer<uint8_t4> input_buffer;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d %d %d %d", input_buffer[gid.x].x, input_buffer[gid.x].y, input_buffer[gid.x].z,
             input_buffer[gid.x].w);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_video_buffer_source_op();

  application->check_output("42 43 44 45");
}

TEST(SlangShaderAppTest, OutputMultiple) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::output("output_buffer_1")]
    [holoscan::alloc(1, 1, 1)]
    RWStructuredBuffer<int> output_buffer_1;

    [holoscan::output("output_buffer_2")]
    [holoscan::alloc(1, 1, 1)]
    RWStructuredBuffer<int> output_buffer_2;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer_1[gid.x] = 4711;
      output_buffer_2[gid.x] = 4712;
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->print_output();
  application->add_sink_op("output_buffer_1");
  application->add_sink_op("output_buffer_2");

  application->check_output("4711\n4712\n");
}

TEST(SlangShaderAppTest, OutputAlloc) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(1, 1, 1)]
    RWStructuredBuffer<int> output_buffer;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer[gid.x] = 4711;
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->print_output();
  application->add_sink_op();

  application->check_output("4711\n");
}

TEST(SlangShaderAppTest, OutputAllocSizeOf) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<int> input_buffer;

    [holoscan::output("output_buffer")]
    [holoscan::alloc::size_of("input_buffer")]
    RWStructuredBuffer<int> output_buffer;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer[gid.x] = input_buffer[gid.x];
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op();
  application->print_output();
  application->add_sink_op();

  application->check_output("42\n");
}

TEST(SlangShaderAppTest, OutputAllocSizeOfWithSwizzle) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<int> input_buffer;

    [holoscan::output("output_buffer")]
    [holoscan::alloc::size_of("input_buffer.2x3")]
    RWStructuredBuffer<int> output_buffer;

    [holoscan::size_of("output_buffer")]
    uint3 size;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer[gid.x] = input_buffer[gid.x];
      printf("%d %d %d\n", size.x, size.y, size.z);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_input_shape({1, 1});
  application->add_source_op();
  application->set_output_shape({3, 1, 2, 1});
  application->print_output();
  application->add_sink_op();

  application->check_output("2 1 3\n42, 0, 0, 0, 0, 0\n");
}

TEST(SlangShaderAppTest, ParameterSizeOf) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(2, 3, 4)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::size_of("output_buffer")]
    uint3 size;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d %d %d\n", size.x, size.y, size.z);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_output_shape({4, 3, 2, 3});
  application->add_sink_op();

  application->check_output("2 3 4\n");
}

TEST(SlangShaderAppTest, ParameterSizeOfError) {
  std::list<std::pair<std::string, std::string>> shader_sources = {
      {R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 5, 6)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::size_of("input_buffer")]
    uint3 size;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
    }
  )",
       "Attribute 'size_of': input 'input_buffer' not found."},

      {R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 5, 6)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::size_of("output_buffer")]
    float size;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
    }
  )",
       "Attribute 'size_of' supports a three component uint32 vector (`uint3`) uniforms only and "
       "cannot be applied to 'size'."},
  };

  for (const auto& [shader_source, expected_error] : shader_sources) {
    auto application = holoscan::make_application<SlangShaderApp>(shader_source);
    application->add_sink_op();

    testing::internal::CaptureStderr();
    bool failed = false;

    try {
      application->run();
    } catch (const std::runtime_error& e) {
      failed = (e.what() != expected_error);
      EXPECT_THAT(e.what(), testing::Eq(expected_error));
    }

    std::string stderr_output = testing::internal::GetCapturedStderr();
    if (failed) {
      std::cout << stderr_output;
    }
  }
}

TEST(SlangShaderAppTest, ParameterStridesOf) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 5, 6)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::strides_of("output_buffer")]
    uint64_t3 strides;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%llu %llu %llu\n", strides.x, strides.y, strides.z);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_output_shape({6, 5, 4, 3});
  application->add_sink_op();

  application->check_output("12 48 240\n");
}

TEST(SlangShaderAppTest, ParameterStridesOfVideoBuffer) {
  const std::string shader_source = R"(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<uint8_t4> input_buffer;

    [holoscan::strides_of("input_buffer")]
    uint64_t3 strides;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%llu %llu %llu\n", strides.x, strides.y, strides.z);
    }
  )";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_video_buffer_source_op();

  // The video source has 4 bytes per pixel, a width of 2 and a height of 3
  // This results in a x stride of 2 * 4 = 8, a y stride of 3 * 8 = 24 and a z stride of also 24.
  application->check_output("8 24 24\n");
}

TEST(SlangShaderAppTest, ParameterStridesOfError) {
  std::list<std::pair<std::string, std::string>> shader_sources = {
      {R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 5, 6)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::strides_of("input_buffer")]
    uint64_t3 strides;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
    }
  )",
       "Attribute 'strides_of': input 'input_buffer' not found."},

      {R"(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 5, 6)]
    RWStructuredBuffer<int3> output_buffer;

    [holoscan::strides_of("output_buffer")]
    float strides;

    [shader("compute")]
    void compute(uint3 gid : SV_DispatchThreadID) {
    }
  )",
       "Attribute 'strides_of' supports a three component uint64 vector (`uint64_t3`) uniforms "
       "only and cannot be applied to 'strides'."},
  };

  for (const auto& [shader_source, expected_error] : shader_sources) {
    auto application = holoscan::make_application<SlangShaderApp>(shader_source);
    application->add_sink_op();

    testing::internal::CaptureStderr();
    bool failed = false;

    try {
      application->run();
    } catch (const std::runtime_error& e) {
      failed = e.what() != expected_error;
      EXPECT_THAT(e.what(), testing::Eq(expected_error));
    }

    std::string stderr_output = testing::internal::GetCapturedStderr();
    if (failed) {
      std::cout << stderr_output;
    }
  }
}

TEST(SlangShaderAppTest, Invocations) {
  const std::string shader_source = R"r(
    import holoscan;

    uint3 get_invocations() {
      __intrinsic_asm "make_uint3(blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z)";
    }

    [shader("compute")]
    [holoscan::invocations(6, 5, 4)]
    void compute(uint3 gid : SV_DispatchThreadID) {
      if (gid.x == 0 && gid.y == 0 && gid.z == 0) {
        uint3 invocations = get_invocations();
        printf("%d %d %d", invocations.x, invocations.y, invocations.z);
      }
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);

  application->check_output("6 5 4");
}

TEST(SlangShaderAppTest, InvocationsSizeOf) {
  const std::string shader_source = R"r(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 3, 2)]
    RWStructuredBuffer<int3> output_buffer;

    uint3 get_invocations() {
      __intrinsic_asm "make_uint3(blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z)";
    }

    [shader("compute")]
    [holoscan::invocations::size_of("output_buffer")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer[gid.x] = gid.x;
      if (gid.x == 0 && gid.y == 0 && gid.z == 0) {
        uint3 invocations = get_invocations();
        printf("%d %d %d\n", invocations.x, invocations.y, invocations.z);
      }
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_output_shape({2, 3, 4, 3});
  application->add_sink_op();

  application->check_output("4 3 2\n");
}

TEST(SlangShaderAppTest, InvocationsSizeOfWithSwizzle) {
  const std::string shader_source = R"r(
    import holoscan;

    [holoscan::output("output_buffer")]
    [holoscan::alloc(4, 3, 2)]
    RWStructuredBuffer<int3> output_buffer;

    uint3 get_invocations() {
      __intrinsic_asm "make_uint3(blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z)";
    }

    [shader("compute")]
    [holoscan::invocations::size_of("output_buffer.zx1")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      output_buffer[gid.x] = gid.x;
      if (gid.x == 0 && gid.y == 0 && gid.z == 0) {
        uint3 invocations = get_invocations();
        printf("%d %d %d\n", invocations.x, invocations.y, invocations.z);
      }
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->set_output_shape({2, 3, 4, 3});
  application->add_sink_op();

  application->check_output("2 4 1\n");
}

TEST(SlangShaderAppTest, InvocationsSizeOfVideoBuffer) {
  const std::string shader_source = R"r(
    import holoscan;

    [holoscan::input("input_buffer")]
    StructuredBuffer<uint8_t4> input_buffer;

    [holoscan::output("output_buffer")]
    [holoscan::alloc::size_of("input_buffer")]
    RWStructuredBuffer<uint8_t4> output_buffer;

    [holoscan::size_of("input_buffer")]
    uint3 size;

    uint3 get_invocations() {
      __intrinsic_asm "make_uint3(blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z)";
    }

    [shader("compute")]
    [holoscan::invocations::size_of("input_buffer")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      int offset = ((gid.z * size.y) + gid.y) * size.x + gid.x;
      output_buffer[offset] = input_buffer[offset];
      if (gid.x == 0 && gid.y == 0 && gid.z == 0) {
        uint3 invocations = get_invocations();
        printf("%d %d %d\n", invocations.x, invocations.y, invocations.z);
      }
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_video_buffer_source_op();
  application->set_output_shape({3, 2, 4});
  application->print_output();
  application->add_sink_op();

  // The video buffer has 4 bytes per pixel, values set to 42, 43, 44, 45, ...
  // The output operator prints 32-bit integers, therefore the output is 757869354, 825241390, ...
  application->check_output(
      "2 3 1\n757869354, 825241390, 892613426, 959985462, 1027357498, 1094729534, 0, 0, 0, 0, 0, "
      "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\n");
}

// Test that the output buffer is the same as the input buffer
TEST(SlangShaderAppTest, Inplace) {
  const std::string shader_source = R"r(
    import holoscan;

    [holoscan::input("input_buffer")]
    [holoscan::output("output_buffer")]
    RWStructuredBuffer<uint8_t4> buffer;

    [shader("compute")]
    [holoscan::invocations::size_of("input_buffer")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      buffer[gid.x] *= 2;
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op();
  application->print_output();
  application->add_sink_op();

  application->check_output("84\n");
}

TEST(SlangShaderAppTest, Zeros) {
  const std::string shader_source = R"r(
    import holoscan;

    [holoscan::input("input_buffer")]
    [holoscan::zeros()]
    RWStructuredBuffer<int> buffer;

    [shader("compute")]
    [holoscan::invocations::size_of("input_buffer")]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d\n", buffer[gid.x]);
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(shader_source);
  application->add_source_op();

  application->check_output("0\n");
}

TEST(SlangShaderAppTest, PreprocessorMacros) {
  const std::string shader_source = R"r(
    import holoscan;

    [shader("compute")]
    [holoscan::invocations(1,1,1)]
    void compute(uint3 gid : SV_DispatchThreadID) {
      printf("%d\n", PREPROCESSOR_MACRO);
    }
  )r";

  auto application = holoscan::make_application<SlangShaderApp>(
      shader_source,
      holoscan::ArgList(
          {holoscan::Arg("preprocessor_macros",
                         std::map<std::string, std::string>{{"PREPROCESSOR_MACRO", "42"}})}));

  application->check_output("42\n");
}
