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
#include "streaming_client.hpp"

using namespace holoscan::ops;

class StreamingClientOpTest : public ::testing::Test {
 protected:
  void SetUp() override { 
    fragment_ = std::make_shared<holoscan::Fragment>(); 
  }

  void TearDown() override { 
    fragment_.reset(); 
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
};

// Test basic operator construction
TEST_F(StreamingClientOpTest, Construction) {
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_op", 
        holoscan::Arg("width", 854),
        holoscan::Arg("height", 480),
        holoscan::Arg("fps", 30),
        holoscan::Arg("server_ip", "127.0.0.1"),
        holoscan::Arg("signaling_port", 48010));
    EXPECT_NE(op, nullptr);
  });
}

// Test operator with minimal parameters
TEST_F(StreamingClientOpTest, MinimalConstruction) {
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>("streaming_client_minimal");
    EXPECT_NE(op, nullptr);
  });
}

// Test operator with various resolutions
TEST_F(StreamingClientOpTest, DifferentResolutions) {
  struct TestCase { uint32_t width, height; };
  std::vector<TestCase> test_cases = {
    {640, 480}, {854, 480}, {1280, 720}, {1920, 1080}
  };

  for (const auto& test_case : test_cases) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_res_test",
          holoscan::Arg("width", test_case.width),
          holoscan::Arg("height", test_case.height));
      EXPECT_NE(op, nullptr);
    }) << "Failed for resolution " << test_case.width << "x" << test_case.height;
  }
}

// Test operator with various frame rates
TEST_F(StreamingClientOpTest, DifferentFrameRates) {
  std::vector<uint32_t> fps_values = {15, 24, 30, 60};

  for (const auto& fps : fps_values) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_fps_test",
          holoscan::Arg("fps", fps));
      EXPECT_NE(op, nullptr);
    }) << "Failed for fps " << fps;
  }
}

// Test operator with different server IPs
TEST_F(StreamingClientOpTest, DifferentServerIPs) {
  std::vector<std::string> server_ips = {
    "127.0.0.1", "192.168.1.100", "10.0.0.1", "localhost"
  };

  for (const auto& ip : server_ips) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_ip_test",
          holoscan::Arg("server_ip", ip));
      EXPECT_NE(op, nullptr);
    }) << "Failed for IP " << ip;
  }
}

// Test operator with different ports
TEST_F(StreamingClientOpTest, DifferentPorts) {
  std::vector<uint16_t> ports = {48010, 8080, 9090, 12345};

  for (const auto& port : ports) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_port_test",
          holoscan::Arg("signaling_port", port));
      EXPECT_NE(op, nullptr);
    }) << "Failed for port " << port;
  }
}

// Test operator with frame handling flags
TEST_F(StreamingClientOpTest, FrameHandlingFlags) {
  struct TestCase { bool receive_frames, send_frames; };
  std::vector<TestCase> test_cases = {
    {true, true}, {true, false}, {false, true}, {false, false}
  };

  for (const auto& test_case : test_cases) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_flags_test",
          holoscan::Arg("receive_frames", test_case.receive_frames),
          holoscan::Arg("send_frames", test_case.send_frames));
      EXPECT_NE(op, nullptr);
    }) << "Failed for rx:" << test_case.receive_frames 
       << ", tx:" << test_case.send_frames;
  }
}

// Test operator setup
TEST_F(StreamingClientOpTest, Setup) {
  auto op = fragment_->make_operator<StreamingClientOp>(
      "streaming_client_setup_test",
      holoscan::Arg("width", 854),
      holoscan::Arg("height", 480));
  
  EXPECT_NO_THROW(op->setup(*op->spec()));
}

// Test invalid parameters (these should not crash during construction)
TEST_F(StreamingClientOpTest, InvalidParameters) {
  // Very large width - should not crash during construction
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_large_width",
        holoscan::Arg("width", 99999));
    EXPECT_NE(op, nullptr);
  });

  // Very large height - should not crash during construction  
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_large_height",
        holoscan::Arg("height", 99999));
    EXPECT_NE(op, nullptr);
  });

  // Very high fps - should not crash during construction
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_high_fps",
        holoscan::Arg("fps", 1000));
    EXPECT_NE(op, nullptr);
  });
}

// Test boundary values
TEST_F(StreamingClientOpTest, BoundaryValues) {
  // Test minimum reasonable values
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_min_values",
        holoscan::Arg("width", 1),
        holoscan::Arg("height", 1),
        holoscan::Arg("fps", 1),
        holoscan::Arg("signaling_port", 1024));
    EXPECT_NE(op, nullptr);
  });

  // Test maximum reasonable values
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_max_values",
        holoscan::Arg("width", 7680),   // 8K width
        holoscan::Arg("height", 4320),  // 8K height
        holoscan::Arg("fps", 120),      // High frame rate
        holoscan::Arg("signaling_port", 65535)); // Maximum port
    EXPECT_NE(op, nullptr);
  });
}

// Test multiple instance creation
TEST_F(StreamingClientOpTest, MultipleInstances) {
  std::vector<std::shared_ptr<StreamingClientOp>> ops;
  
  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_" + std::to_string(i),
          holoscan::Arg("width", 640 + i*100),
          holoscan::Arg("height", 480 + i*50),
          holoscan::Arg("fps", 30 + i*10));
      EXPECT_NE(op, nullptr);
      ops.push_back(op);
    }) << "Failed to create instance " << i;
  }
  
  // Verify all instances are unique
  EXPECT_EQ(ops.size(), 3);
  for (size_t i = 0; i < ops.size(); ++i) {
    for (size_t j = i + 1; j < ops.size(); ++j) {
      EXPECT_NE(ops[i], ops[j]) << "Instances " << i << " and " << j << " should be different";
    }
  }
}

// Test with string parameters
TEST_F(StreamingClientOpTest, StringParameters) {
  // Test empty server IP (construction should succeed, errors might occur later)
  EXPECT_NO_THROW({
    auto op = fragment_->make_operator<StreamingClientOp>(
        "streaming_client_empty_ip",
        holoscan::Arg("server_ip", ""));
    EXPECT_NE(op, nullptr);
  });

  // Test various valid server IP formats
  std::vector<std::string> valid_ips = {
    "127.0.0.1", "0.0.0.0", "255.255.255.255", "localhost", "example.com"
  };
  
  for (const auto& ip : valid_ips) {
    EXPECT_NO_THROW({
      auto op = fragment_->make_operator<StreamingClientOp>(
          "streaming_client_valid_ip",
          holoscan::Arg("server_ip", ip));
      EXPECT_NE(op, nullptr);
    }) << "Failed for IP: " << ip;
  }
}
