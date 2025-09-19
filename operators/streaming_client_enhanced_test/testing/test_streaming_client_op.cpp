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

#include <gtest/gtest.h>
#include <memory>

#include "../streaming_client.hpp"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class StreamingClientOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = holoscan::make_fragment<holoscan::Fragment>();
    
    // Create the operator with default parameters
    streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
        "test_streaming_client",
        holoscan::Arg("width") = 640u,
        holoscan::Arg("height") = 480u,
        holoscan::Arg("fps") = 30u,
        holoscan::Arg("server_ip") = std::string("127.0.0.1"),
        holoscan::Arg("signaling_port") = 48010,
        holoscan::Arg("send_frames") = false,  // Disable for unit testing
        holoscan::Arg("receive_frames") = false  // Disable for unit testing
    );
  }

  void TearDown() override {
    streaming_client_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingClientOp> streaming_client_op_;
};

// Test 1: Operator Initialization
TEST_F(StreamingClientOpTest, OperatorInitialization) {
  ASSERT_NE(streaming_client_op_, nullptr);
  EXPECT_EQ(streaming_client_op_->name(), "test_streaming_client");
}

// Test 2: Parameter Configuration
TEST_F(StreamingClientOpTest, ParameterConfiguration) {
  // The operator should have proper default values
  // Note: We can't easily access private parameters in this test structure,
  // so we verify the operator was created successfully with the parameters
  EXPECT_NO_THROW({
    // This verifies the operator can be created with valid parameters
    auto test_op = fragment_->make_operator<StreamingClientOp>(
        "test_param",
        holoscan::Arg("width") = 1920u,
        holoscan::Arg("height") = 1080u,
        holoscan::Arg("fps") = 60u
    );
  });
}

// Test 3: Invalid Parameter Handling  
TEST_F(StreamingClientOpTest, InvalidParameterHandling) {
  // Test with invalid width (0)
  EXPECT_NO_THROW({
    auto test_op = fragment_->make_operator<StreamingClientOp>(
        "test_invalid",
        holoscan::Arg("width") = 0u,  // Invalid width
        holoscan::Arg("height") = 480u,
        holoscan::Arg("send_frames") = false,
        holoscan::Arg("receive_frames") = false
    );
    // The operator should handle this gracefully during initialization
  });
}

// Test 4: Server IP Configuration
TEST_F(StreamingClientOpTest, ServerIPConfiguration) {
  EXPECT_NO_THROW({
    auto test_op = fragment_->make_operator<StreamingClientOp>(
        "test_ip",
        holoscan::Arg("server_ip") = std::string("192.168.1.100"),
        holoscan::Arg("signaling_port") = 8080,
        holoscan::Arg("send_frames") = false,
        holoscan::Arg("receive_frames") = false
    );
  });
}

// Test 5: Port Range Validation
TEST_F(StreamingClientOpTest, PortRangeValidation) {
  // Test valid port
  EXPECT_NO_THROW({
    auto test_op = fragment_->make_operator<StreamingClientOp>(
        "test_port_valid",
        holoscan::Arg("signaling_port") = 8080,
        holoscan::Arg("send_frames") = false,
        holoscan::Arg("receive_frames") = false
    );
  });

  // Test edge case ports
  EXPECT_NO_THROW({
    auto test_op = fragment_->make_operator<StreamingClientOp>(
        "test_port_edge",
        holoscan::Arg("signaling_port") = 65535,  // Max valid port
        holoscan::Arg("send_frames") = false,
        holoscan::Arg("receive_frames") = false
    );
  });
}

// Test 6: Frame Rate Configuration
TEST_F(StreamingClientOpTest, FrameRateConfiguration) {
  // Test various frame rates
  std::vector<uint32_t> test_fps = {1, 30, 60, 120};
  
  for (auto fps : test_fps) {
    EXPECT_NO_THROW({
      auto test_op = fragment_->make_operator<StreamingClientOp>(
          "test_fps_" + std::to_string(fps),
          holoscan::Arg("fps") = fps,
          holoscan::Arg("send_frames") = false,
          holoscan::Arg("receive_frames") = false
      );
    }) << "Failed with FPS: " << fps;
  }
}

// Test 7: Resolution Configuration
TEST_F(StreamingClientOpTest, ResolutionConfiguration) {
  // Test common resolutions
  std::vector<std::pair<uint32_t, uint32_t>> resolutions = {
    {640, 480},    // VGA
    {1280, 720},   // HD
    {1920, 1080},  // Full HD
    {3840, 2160}   // 4K
  };
  
  for (const auto& [width, height] : resolutions) {
    EXPECT_NO_THROW({
      auto test_op = fragment_->make_operator<StreamingClientOp>(
          "test_res_" + std::to_string(width) + "x" + std::to_string(height),
          holoscan::Arg("width") = width,
          holoscan::Arg("height") = height,
          holoscan::Arg("send_frames") = false,
          holoscan::Arg("receive_frames") = false
      );
    }) << "Failed with resolution: " << width << "x" << height;
  }
}

// Test 8: Boolean Parameter Configuration
TEST_F(StreamingClientOpTest, BooleanParameterConfiguration) {
  // Test all combinations of send_frames and receive_frames
  std::vector<std::pair<bool, bool>> combinations = {
    {false, false},  // No streaming (good for unit tests)
    {true, false},   // Send only
    {false, true},   // Receive only  
    {true, true}     // Bidirectional
  };
  
  for (const auto& [send, receive] : combinations) {
    EXPECT_NO_THROW({
      auto test_op = fragment_->make_operator<StreamingClientOp>(
          "test_bool_" + std::to_string(send) + "_" + std::to_string(receive),
          holoscan::Arg("send_frames") = send,
          holoscan::Arg("receive_frames") = receive
      );
    }) << "Failed with send=" << send << ", receive=" << receive;
  }
}

}  // namespace holoscan::ops

// Main function for running the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
