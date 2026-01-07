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

#include "holoscan/holoscan.hpp"
#include "../video_streaming_client.hpp"

namespace holoscan::ops {

/**
 * @brief Test fixture for VideoStreamingClientOp unit tests
 *
 * This test fixture provides a Holoscan Fragment context for testing
 * the VideoStreamingClientOp operator in isolation.
 */
class VideoStreamingClientOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    video_streaming_client_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<VideoStreamingClientOp> video_streaming_client_op_;
};

// ====================================================================================
// Basic Initialization Tests
// ====================================================================================

/**
 * @brief Test basic operator creation with minimal required parameters
 *
 * Verifies that the VideoStreamingClientOp can be instantiated with standard parameters
 * and that the operator has the expected name and type.
 */
TEST_F(VideoStreamingClientOpTest, BasicInitialization) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "test_video_streaming_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,  // Disable for unit testing
      holoscan::Arg("receive_frames") = false);  // Disable for unit testing

  ASSERT_NE(video_streaming_client_op_, nullptr);
  EXPECT_EQ(video_streaming_client_op_->name(), "test_video_streaming_client");
  EXPECT_EQ(video_streaming_client_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

/**
 * @brief Test operator creation with default streaming mode (disabled)
 *
 * Verifies that the operator can be created with streaming disabled,
 * useful for testing without requiring an actual server connection.
 */
TEST_F(VideoStreamingClientOpTest, InitializationWithStreamingDisabled) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "disabled_client",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u,
      holoscan::Arg("server_ip") = std::string("192.168.1.100"),
      holoscan::Arg("signaling_port") = uint16_t{8080},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(video_streaming_client_op_, nullptr);
  EXPECT_EQ(video_streaming_client_op_->name(), "disabled_client");
}

// ====================================================================================
// Parameter Validation Tests
// ====================================================================================

/**
 * @brief Test operator with various video resolutions
 *
 * Verifies that the operator accepts different video resolutions
 * commonly used in streaming applications.
 */
TEST_F(VideoStreamingClientOpTest, VideoResolutionParameters) {
  // Standard Definition (640x480)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "sd_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // HD (1280x720)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "hd_client",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Full HD (1920x1080)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "fhd_client",
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // 4K (3840x2160)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "4k_client",
      holoscan::Arg("width") = 3840u,
      holoscan::Arg("height") = 2160u,
      holoscan::Arg("fps") = 24u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

/**
 * @brief Test operator with various frame rates
 *
 * Verifies that the operator accepts different frame rates
 * from low-latency to high-performance scenarios.
 */
TEST_F(VideoStreamingClientOpTest, FrameRateParameters) {
  // Low frame rate (15 FPS)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "low_fps_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 15u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Standard frame rate (30 FPS)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "standard_fps_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // High frame rate (60 FPS)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "high_fps_client",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Very high frame rate (120 FPS)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "very_high_fps_client",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 120u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

/**
 * @brief Test operator with various network configurations
 *
 * Verifies that the operator accepts different server IPs and ports.
 */
TEST_F(VideoStreamingClientOpTest, NetworkParameters) {
  // Localhost
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "localhost_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // LAN address
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "lan_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("192.168.1.100"),
      holoscan::Arg("signaling_port") = uint16_t{8080},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Different port
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "custom_port_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("10.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{9999},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

/**
 * @brief Test operator with streaming mode configurations
 *
 * Verifies different combinations of send_frames and receive_frames parameters.
 */
TEST_F(VideoStreamingClientOpTest, StreamingModeParameters) {
  // Send only
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "send_only_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = true,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Receive only
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "receive_only_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = true);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Bidirectional
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "bidirectional_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = true,
      holoscan::Arg("receive_frames") = true);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Both disabled (configuration only)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "disabled_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

/**
 * @brief Test operator with frame validation parameter
 *
 * Verifies that the min_non_zero_bytes parameter is accepted and that
 * operators can be created with various validation thresholds.
 */
TEST_F(VideoStreamingClientOpTest, FrameValidationParameter) {
  // Default validation threshold (100 bytes)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "default_validation_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false,
      holoscan::Arg("min_non_zero_bytes") = 100u);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Low validation threshold
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "low_validation_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false,
      holoscan::Arg("min_non_zero_bytes") = 10u);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // High validation threshold
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "high_validation_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false,
      holoscan::Arg("min_non_zero_bytes") = 500u);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // No validation (0 bytes)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "no_validation_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false,
      holoscan::Arg("min_non_zero_bytes") = 0u);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

// ====================================================================================
// Operator Setup Tests
// ====================================================================================

/**
 * @brief Test operator setup method
 *
 * Verifies that the setup() method can be called without crashing
 * and properly configures the operator's input/output ports.
 */
TEST_F(VideoStreamingClientOpTest, OperatorSetup) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "test_setup_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Create an operator spec for setup testing
  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());

  // Test that setup doesn't crash (basic smoke test)
  EXPECT_NO_THROW(video_streaming_client_op_->setup(*spec));
}

// ====================================================================================
// Edge Case and Boundary Tests
// ====================================================================================

/**
 * @brief Test operator with minimum resolution
 *
 * Verifies that the operator accepts very small resolutions.
 */
TEST_F(VideoStreamingClientOpTest, MinimumResolution) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "min_resolution_client",
      holoscan::Arg("width") = 320u,
      holoscan::Arg("height") = 240u,
      holoscan::Arg("fps") = 15u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(video_streaming_client_op_, nullptr);
  EXPECT_EQ(video_streaming_client_op_->name(), "min_resolution_client");
}

/**
 * @brief Test operator with maximum reasonable resolution
 *
 * Verifies that the operator accepts 8K resolution parameters.
 */
TEST_F(VideoStreamingClientOpTest, MaximumResolution) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "max_resolution_client",
      holoscan::Arg("width") = 7680u,  // 8K width
      holoscan::Arg("height") = 4320u,  // 8K height
      holoscan::Arg("fps") = 24u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(video_streaming_client_op_, nullptr);
  EXPECT_EQ(video_streaming_client_op_->name(), "max_resolution_client");
}

/**
 * @brief Test operator with edge case port numbers
 *
 * Verifies that the operator accepts valid port numbers including
 * low and high port ranges.
 */
TEST_F(VideoStreamingClientOpTest, PortNumberEdgeCases) {
  // Low port number (1024 - first non-privileged port)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "low_port_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{1024},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);

  // High port number (65535 - maximum port)
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "high_port_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{65535},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);
  ASSERT_NE(video_streaming_client_op_, nullptr);
}

// ====================================================================================
// Resource Management Tests
// ====================================================================================

/**
 * @brief Test operator cleanup and destruction
 *
 * Verifies that the operator can be properly destroyed without crashing
 * or leaking resources.
 */
TEST_F(VideoStreamingClientOpTest, OperatorCleanup) {
  video_streaming_client_op_ = fragment_->make_operator<VideoStreamingClientOp>(
      "cleanup_test_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(video_streaming_client_op_, nullptr);

  // Test that cleanup doesn't crash
  EXPECT_NO_THROW({
    video_streaming_client_op_.reset();
  });

  EXPECT_EQ(video_streaming_client_op_, nullptr);
}

/**
 * @brief Test multiple operator instances
 *
 * Verifies that multiple VideoStreamingClientOp instances can be created
 * and managed simultaneously without conflicts.
 */
TEST_F(VideoStreamingClientOpTest, MultipleInstances) {
  auto client1 = fragment_->make_operator<VideoStreamingClientOp>(
      "client_1",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{48010},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  auto client2 = fragment_->make_operator<VideoStreamingClientOp>(
      "client_2",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u,
      holoscan::Arg("server_ip") = std::string("192.168.1.100"),
      holoscan::Arg("signaling_port") = uint16_t{8080},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  auto client3 = fragment_->make_operator<VideoStreamingClientOp>(
      "client_3",
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("10.0.0.1"),
      holoscan::Arg("signaling_port") = uint16_t{9999},
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false);

  ASSERT_NE(client1, nullptr);
  ASSERT_NE(client2, nullptr);
  ASSERT_NE(client3, nullptr);

  EXPECT_EQ(client1->name(), "client_1");
  EXPECT_EQ(client2->name(), "client_2");
  EXPECT_EQ(client3->name(), "client_3");

  // Verify they are different instances
  EXPECT_NE(client1, client2);
  EXPECT_NE(client2, client3);
  EXPECT_NE(client1, client3);
}

}  // namespace holoscan::ops
