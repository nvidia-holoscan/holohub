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

#include "test_utilities.hpp"

#include <random>
#include <algorithm>
#include <cmath>

namespace holoscan::ops::streaming_server_enhanced::testing {

// TestFrameGenerator implementation

std::vector<uint8_t> TestFrameGenerator::generateTestFrame(uint32_t width, uint32_t height,
                                                            const std::string& pattern,
                                                            uint8_t color_value) {
  std::vector<uint8_t> frame_data(width * height * 3);  // BGR format
  
  if (pattern == "solid") {
    std::fill(frame_data.begin(), frame_data.end(), color_value);
  } else if (pattern == "gradient") {
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        size_t idx = (y * width + x) * 3;
        // Create RGB gradient
        frame_data[idx] = static_cast<uint8_t>((x * 255) / width);     // B
        frame_data[idx + 1] = static_cast<uint8_t>((y * 255) / height); // G
        frame_data[idx + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height)); // R
      }
    }
  } else if (pattern == "checkerboard") {
    uint32_t block_size = std::max(1u, std::min(width, height) / 16);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        size_t idx = (y * width + x) * 3;
        uint8_t value = ((x / block_size) + (y / block_size)) % 2 ? 255 : 0;
        frame_data[idx] = value;     // B
        frame_data[idx + 1] = value; // G
        frame_data[idx + 2] = value; // R
      }
    }
  } else if (pattern == "noise") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    
    std::generate(frame_data.begin(), frame_data.end(), [&]() { return dis(gen); });
  } else {
    // Default: fill with test pattern (horizontal stripes)
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        size_t idx = (y * width + x) * 3;
        uint8_t value = static_cast<uint8_t>((y * 255) / height);
        frame_data[idx] = value;     // B
        frame_data[idx + 1] = value; // G
        frame_data[idx + 2] = value; // R
      }
    }
  }
  
  return frame_data;
}

std::vector<std::vector<uint8_t>> TestFrameGenerator::generateTestFrameSequence(
    size_t count, uint32_t width, uint32_t height, const std::string& pattern) {
  std::vector<std::vector<uint8_t>> sequence;
  sequence.reserve(count);
  
  for (size_t i = 0; i < count; ++i) {
    if (pattern == "animated_gradient") {
      // Create animated gradient by shifting colors
      uint8_t offset = static_cast<uint8_t>((i * 32) % 256);
      sequence.push_back(generateTestFrame(width, height, "gradient", offset));
    } else {
      sequence.push_back(generateTestFrame(width, height, pattern));
    }
  }
  
  return sequence;
}

bool TestFrameGenerator::validateFrameProperties(const std::vector<uint8_t>& frame_data,
                                                  uint32_t expected_width,
                                                  uint32_t expected_height,
                                                  uint32_t channels) {
  size_t expected_size = static_cast<size_t>(expected_width) * expected_height * channels;
  return frame_data.size() == expected_size;
}

// TestTensorUtils implementation

std::shared_ptr<holoscan::Tensor> TestTensorUtils::createTensorFromFrame(
    const std::vector<uint8_t>& frame_data, uint32_t width, uint32_t height) {
  // Note: This is a simplified mock implementation
  // In a real implementation, you would create an actual Holoscan tensor
  // For now, we return nullptr as a placeholder
  return nullptr;
}

std::shared_ptr<holoscan::TensorMap> TestTensorUtils::createTensorMap(
    const std::string& tensor_name, std::shared_ptr<holoscan::Tensor> tensor) {
  // Note: This is a simplified mock implementation
  // In a real implementation, you would create an actual tensor map
  // For now, we return nullptr as a placeholder
  return nullptr;
}

bool TestTensorUtils::validateTensorProperties(const holoscan::Tensor& tensor,
                                                const std::vector<int64_t>& expected_shape,
                                                holoscan::nvidia::gxf::PrimitiveType expected_dtype) {
  // Note: This is a simplified mock implementation
  // In a real implementation, you would validate actual tensor properties
  return true;
}

// StreamingServerTestFixture implementation

void StreamingServerTestFixture::SetUp() {
  fragment_ = std::make_shared<MockFragment>();
  executor_ = std::make_shared<MockExecutor>(fragment_.get());
  execution_context_ = std::make_shared<MockExecutionContext>(fragment_.get(), executor_.get());
  
  // Note: Input/Output contexts need an operator instance
  // For now, we'll set them to nullptr and create them in individual tests
  input_context_ = nullptr;
  output_context_ = nullptr;
  
  // Set up default test configuration
  test_config_.width = 854;
  test_config_.height = 480;
  test_config_.fps = 30;
  test_config_.port = 48010;
  test_config_.server_name = "TestStreamingServer";
  test_config_.enable_debug = false;
}

void StreamingServerTestFixture::TearDown() {
  output_context_.reset();
  input_context_.reset();
  execution_context_.reset();
  executor_.reset();
  fragment_.reset();
}

}  // namespace holoscan::ops::streaming_server_enhanced::testing
