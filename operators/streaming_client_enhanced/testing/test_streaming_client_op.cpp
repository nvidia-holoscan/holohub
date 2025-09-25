// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <memory>

#include "holoscan/holoscan.hpp"
#include "streaming_client.hpp"

namespace holoscan::ops {

class StreamingClientOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = holoscan::make_fragment<holoscan::Fragment>();
  }

  void TearDown() override {
    streaming_client_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingClientOp> streaming_client_op_;
};

// Test basic operator creation and initialization
TEST_F(StreamingClientOpTest, BasicInitialization) {
  // Create the operator with minimal parameters
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "test_streaming_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = 48010,
      holoscan::Arg("send_frames") = false,  // Disable for unit testing
      holoscan::Arg("receive_frames") = false);  // Disable for unit testing

  // Verify the operator was created successfully
  ASSERT_NE(streaming_client_op_, nullptr);

  // Verify the operator has the expected name
  EXPECT_EQ(streaming_client_op_->name(), "test_streaming_client");

  // Verify operator type
  EXPECT_EQ(streaming_client_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

// Test operator creation with different video parameters
TEST_F(StreamingClientOpTest, ParameterValidation) {
  // Test with HD resolution
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "test_hd_client",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u,
      holoscan::Arg("server_ip") = std::string("192.168.1.100"),
      holoscan::Arg("signaling_port") = 8080,
      holoscan::Arg("send_frames") = true,
      holoscan::Arg("receive_frames") = false,
      holoscan::Arg("min_non_zero_bytes") = 100u);

  ASSERT_NE(streaming_client_op_, nullptr);
  EXPECT_EQ(streaming_client_op_->name(), "test_hd_client");
}

// Test operator setup method
TEST_F(StreamingClientOpTest, OperatorSetup) {
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "test_setup_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = 48010,
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false
);

  ASSERT_NE(streaming_client_op_, nullptr);

  // Create an operator spec for setup testing
  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());

  // Test that setup doesn't crash (basic smoke test)
  EXPECT_NO_THROW(streaming_client_op_->setup(*spec));
}

// Test operator with minimum parameters
TEST_F(StreamingClientOpTest, MinimalParameters) {
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "minimal_client",
      holoscan::Arg("width") = 320u,
      holoscan::Arg("height") = 240u,
      holoscan::Arg("fps") = 15u,
      holoscan::Arg("server_ip") = std::string("localhost"),
      holoscan::Arg("signaling_port") = 3000,
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false
);

  ASSERT_NE(streaming_client_op_, nullptr);
  EXPECT_EQ(streaming_client_op_->name(), "minimal_client");
}

// Test operator parameter edge cases
TEST_F(StreamingClientOpTest, ParameterEdgeCases) {
  // Test with maximum reasonable values
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "max_params_client",
      holoscan::Arg("width") = 3840u,       // 4K width
      holoscan::Arg("height") = 2160u,      // 4K height
      holoscan::Arg("fps") = 120u,          // High FPS
      holoscan::Arg("server_ip") = std::string("255.255.255.255"),
      holoscan::Arg("signaling_port") = 65535,  // Max port
      holoscan::Arg("send_frames") = true,
      holoscan::Arg("receive_frames") = true,
      holoscan::Arg("min_non_zero_bytes") = 1000u
);

  ASSERT_NE(streaming_client_op_, nullptr);
  EXPECT_EQ(streaming_client_op_->name(), "max_params_client");
}

// Test operator destruction and cleanup
TEST_F(StreamingClientOpTest, OperatorCleanup) {
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "cleanup_test_client",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u,
      holoscan::Arg("server_ip") = std::string("127.0.0.1"),
      holoscan::Arg("signaling_port") = 48010,
      holoscan::Arg("send_frames") = false,
      holoscan::Arg("receive_frames") = false
);

  ASSERT_NE(streaming_client_op_, nullptr);

  // Test that cleanup doesn't crash
  EXPECT_NO_THROW({
    streaming_client_op_.reset();
  });

  EXPECT_EQ(streaming_client_op_, nullptr);
}

}  // namespace holoscan::ops
