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

#ifndef HOLOSCAN_OPERATORS_STREAMING_SERVER_ENHANCED_TESTING_TEST_UTILITIES_HPP
#define HOLOSCAN_OPERATORS_STREAMING_SERVER_ENHANCED_TESTING_TEST_UTILITIES_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/executor.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace holoscan::ops::streaming_server_enhanced::testing {

/**
 * @brief Test configuration struct for streaming server tests
 */
struct TestConfig {
  uint32_t width = 854;
  uint32_t height = 480;
  uint32_t fps = 30;
  uint16_t port = 48010;
  std::string server_name = "TestStreamingServer";
  bool enable_debug = false;
};

/**
 * @brief Mock fragment for testing
 */
class MockFragment : public holoscan::Fragment {
 public:
  MockFragment() : holoscan::Fragment() {}
  
  void compose() override {
    // Mock implementation - does nothing
  }
};

/**
 * @brief Mock executor for testing
 */
class MockExecutor : public holoscan::Executor {
 public:
  MockExecutor(holoscan::Fragment* fragment) : holoscan::Executor(fragment) {}
  
  void run(holoscan::OperatorGraph& graph) override {
    // Mock implementation - does nothing
  }
};

/**
 * @brief Mock operator spec for testing
 */
class MockOperatorSpec : public holoscan::OperatorSpec {
 public:
  MockOperatorSpec(holoscan::Fragment* fragment) : holoscan::OperatorSpec(fragment) {}
};

/**
 * @brief Mock execution context for testing
 */
class MockExecutionContext : public holoscan::ExecutionContext {
 public:
  MockExecutionContext(holoscan::Fragment* fragment, 
                       holoscan::Executor* executor)
    : holoscan::ExecutionContext(fragment, executor) {}
};

/**
 * @brief Mock input context for testing
 */
class MockInputContext : public holoscan::InputContext {
 public:
  MockInputContext(holoscan::ExecutionContext* execution_context,
                   holoscan::Operator* op)
    : holoscan::InputContext(execution_context, op) {}
};

/**
 * @brief Mock output context for testing
 */
class MockOutputContext : public holoscan::OutputContext {
 public:
  MockOutputContext(holoscan::ExecutionContext* execution_context,
                    holoscan::Operator* op)
    : holoscan::OutputContext(execution_context, op) {}
};

/**
 * @brief Test frame data generator
 */
class TestFrameGenerator {
 public:
  /**
   * @brief Generate a test frame with specified pattern
   * @param width Frame width
   * @param height Frame height
   * @param pattern Pattern type: "gradient", "solid", "checkerboard", "noise"
   * @param color_value Base color value for solid pattern
   * @return Vector containing frame data (BGR format)
   */
  static std::vector<uint8_t> generateTestFrame(uint32_t width, uint32_t height,
                                                const std::string& pattern = "gradient",
                                                uint8_t color_value = 128);

  /**
   * @brief Generate a sequence of test frames
   * @param count Number of frames to generate
   * @param width Frame width
   * @param height Frame height
   * @param pattern Pattern type
   * @return Vector of frame data vectors
   */
  static std::vector<std::vector<uint8_t>> generateTestFrameSequence(
      size_t count, uint32_t width, uint32_t height,
      const std::string& pattern = "gradient");

  /**
   * @brief Validate frame properties
   * @param frame_data Frame data to validate
   * @param expected_width Expected frame width
   * @param expected_height Expected frame height
   * @param channels Number of color channels (default: 3 for BGR)
   * @return True if frame properties match expected values
   */
  static bool validateFrameProperties(const std::vector<uint8_t>& frame_data,
                                      uint32_t expected_width,
                                      uint32_t expected_height,
                                      uint32_t channels = 3);
};

/**
 * @brief Test tensor utilities
 */
class TestTensorUtils {
 public:
  /**
   * @brief Create a mock tensor from frame data
   * @param frame_data Frame data (BGR format)
   * @param width Frame width
   * @param height Frame height
   * @return Shared pointer to created tensor
   */
  static std::shared_ptr<holoscan::Tensor> createTensorFromFrame(
      const std::vector<uint8_t>& frame_data,
      uint32_t width, uint32_t height);

  /**
   * @brief Create a mock tensor map
   * @param tensor_name Name for the tensor in the map
   * @param tensor Tensor to add to the map
   * @return Shared pointer to created tensor map
   */
  static std::shared_ptr<holoscan::TensorMap> createTensorMap(
      const std::string& tensor_name,
      std::shared_ptr<holoscan::Tensor> tensor);

  /**
   * @brief Validate tensor properties
   * @param tensor Tensor to validate
   * @param expected_shape Expected tensor shape
   * @param expected_dtype Expected data type
   * @return True if tensor properties match expected values
   */
  static bool validateTensorProperties(const holoscan::Tensor& tensor,
                                       const std::vector<int64_t>& expected_shape,
                                       holoscan::nvidia::gxf::PrimitiveType expected_dtype);
};

/**
 * @brief Base test fixture for streaming server tests
 */
class StreamingServerTestFixture : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  std::shared_ptr<MockFragment> fragment_;
  std::shared_ptr<MockExecutor> executor_;
  std::shared_ptr<MockExecutionContext> execution_context_;
  std::shared_ptr<MockInputContext> input_context_;
  std::shared_ptr<MockOutputContext> output_context_;
  TestConfig test_config_;
};

/**
 * @brief Test assertions and matchers
 */
#define ASSERT_FRAME_VALID(frame_data, width, height, channels) \
  ASSERT_TRUE(TestFrameGenerator::validateFrameProperties(frame_data, width, height, channels)) \
      << "Frame validation failed: expected " << width << "x" << height \
      << " with " << channels << " channels, got " << frame_data.size() << " bytes"

#define EXPECT_FRAME_VALID(frame_data, width, height, channels) \
  EXPECT_TRUE(TestFrameGenerator::validateFrameProperties(frame_data, width, height, channels)) \
      << "Frame validation failed: expected " << width << "x" << height \
      << " with " << channels << " channels, got " << frame_data.size() << " bytes"

#define ASSERT_TENSOR_VALID(tensor, shape, dtype) \
  ASSERT_TRUE(TestTensorUtils::validateTensorProperties(tensor, shape, dtype)) \
      << "Tensor validation failed"

#define EXPECT_TENSOR_VALID(tensor, shape, dtype) \
  EXPECT_TRUE(TestTensorUtils::validateTensorProperties(tensor, shape, dtype)) \
      << "Tensor validation failed"

}  // namespace holoscan::ops::streaming_server_enhanced::testing

#endif  // HOLOSCAN_OPERATORS_STREAMING_SERVER_ENHANCED_TESTING_TEST_UTILITIES_HPP
