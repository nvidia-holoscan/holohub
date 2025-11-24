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
#include "../video_streaming_server_resource.hpp"
#include "../video_streaming_server_upstream_op.hpp"
#include "../video_streaming_server_downstream_op.hpp"

namespace holoscan::ops {

// ====================================================================================
// StreamingServerResource Tests
// ====================================================================================

/**
 * @brief Test fixture for StreamingServerResource unit tests
 *
 * This test fixture provides a Holoscan Fragment context for testing
 * the StreamingServerResource in isolation.
 */
class StreamingServerResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    resource_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingServerResource> resource_;
};

/**
 * @brief Test basic resource creation with default parameters
 *
 * Verifies that the StreamingServerResource can be instantiated with default
 * parameters and that the resource has the expected name.
 */
TEST_F(StreamingServerResourceTest, BasicResourceInitialization) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_streaming_server_resource");

  ASSERT_NE(resource_, nullptr);
  EXPECT_EQ(resource_->name(), "test_streaming_server_resource");
}

/**
 * @brief Test resource creation with custom parameters
 *
 * Verifies that the resource accepts custom configuration parameters.
 */
TEST_F(StreamingServerResourceTest, ResourceInitializationWithCustomParameters) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "custom_resource",
      holoscan::Arg("port") = uint16_t{8080},
      holoscan::Arg("server_name") = std::string("CustomServer"),
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = uint16_t{60});

  ASSERT_NE(resource_, nullptr);
  EXPECT_EQ(resource_->name(), "custom_resource");
}

/**
 * @brief Test resource with various configuration parameters
 *
 * Verifies that the resource accepts different video and network configurations.
 */
TEST_F(StreamingServerResourceTest, ResourceConfigurationParameters) {
  // Standard configuration
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "standard_config",
      holoscan::Arg("port") = uint16_t{48010},
      holoscan::Arg("width") = 854u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = uint16_t{30});
  ASSERT_NE(resource_, nullptr);

  // HD configuration
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "hd_config",
      holoscan::Arg("port") = uint16_t{48020},
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = uint16_t{60});
  ASSERT_NE(resource_, nullptr);

  // 4K configuration
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "4k_config",
      holoscan::Arg("port") = uint16_t{48030},
      holoscan::Arg("width") = 3840u,
      holoscan::Arg("height") = 2160u,
      holoscan::Arg("fps") = uint16_t{30});
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource with different port configurations
 *
 * Verifies that the resource accepts various port numbers.
 */
TEST_F(StreamingServerResourceTest, ResourcePortConfiguration) {
  // Default port
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "default_port",
      holoscan::Arg("port") = uint16_t{48010});
  ASSERT_NE(resource_, nullptr);

  // Custom port
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "custom_port",
      holoscan::Arg("port") = uint16_t{9999});
  ASSERT_NE(resource_, nullptr);

  // High port number
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "high_port",
      holoscan::Arg("port") = uint16_t{65000});
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource multi-instance mode
 *
 * Verifies that the resource supports multi-instance configuration.
 */
TEST_F(StreamingServerResourceTest, ResourceMultiInstanceMode) {
  // Single instance mode (default)
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "single_instance",
      holoscan::Arg("is_multi_instance") = false);
  ASSERT_NE(resource_, nullptr);

  // Multi-instance mode
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "multi_instance",
      holoscan::Arg("is_multi_instance") = true);
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource upstream/downstream configuration
 *
 * Verifies that the resource can enable/disable upstream and downstream.
 */
TEST_F(StreamingServerResourceTest, ResourceUpstreamDownstreamConfiguration) {
  // Both enabled (default)
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "both_enabled",
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = true);
  ASSERT_NE(resource_, nullptr);

  // Upstream only
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "upstream_only",
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = false);
  ASSERT_NE(resource_, nullptr);

  // Downstream only
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "downstream_only",
      holoscan::Arg("enable_upstream") = false,
      holoscan::Arg("enable_downstream") = true);
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource cleanup and destruction
 *
 * Verifies that the resource can be properly destroyed without crashing.
 */
TEST_F(StreamingServerResourceTest, ResourceCleanup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "cleanup_test");

  ASSERT_NE(resource_, nullptr);

  // Test that cleanup doesn't crash
  EXPECT_NO_THROW({
    resource_.reset();
  });

  EXPECT_EQ(resource_, nullptr);
}

/**
 * @brief Test multiple resource instances
 *
 * Verifies that multiple StreamingServerResource instances can be created
 * and managed simultaneously without conflicts.
 */
TEST_F(StreamingServerResourceTest, MultipleResourceInstances) {
  auto resource1 = fragment_->make_resource<StreamingServerResource>(
      "resource_1",
      holoscan::Arg("port") = uint16_t{48010});

  auto resource2 = fragment_->make_resource<StreamingServerResource>(
      "resource_2",
      holoscan::Arg("port") = uint16_t{48020});

  auto resource3 = fragment_->make_resource<StreamingServerResource>(
      "resource_3",
      holoscan::Arg("port") = uint16_t{48030});

  ASSERT_NE(resource1, nullptr);
  ASSERT_NE(resource2, nullptr);
  ASSERT_NE(resource3, nullptr);

  EXPECT_EQ(resource1->name(), "resource_1");
  EXPECT_EQ(resource2->name(), "resource_2");
  EXPECT_EQ(resource3->name(), "resource_3");

  // Verify they are different instances
  EXPECT_NE(resource1, resource2);
  EXPECT_NE(resource2, resource3);
  EXPECT_NE(resource1, resource3);
}

// ====================================================================================
// StreamingServerUpstreamOp Tests
// ====================================================================================

/**
 * @brief Test fixture for StreamingServerUpstreamOp unit tests
 *
 * This test fixture provides a Holoscan Fragment context for testing
 * the StreamingServerUpstreamOp operator.
 */
class StreamingServerUpstreamOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    upstream_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingServerUpstreamOp> upstream_op_;
};

/**
 * @brief Test basic upstream operator creation
 *
 * Verifies that the StreamingServerUpstreamOp can be instantiated with a
 * shared resource and standard parameters.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpBasicInitialization) {
  // Create shared resource
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "upstream_resource",
      holoscan::Arg("port") = uint16_t{48010});

  // Create upstream operator
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "test_upstream_op",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(upstream_op_, nullptr);
  EXPECT_EQ(upstream_op_->name(), "test_upstream_op");
  EXPECT_EQ(upstream_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

/**
 * @brief Test upstream operator with custom parameters
 *
 * Verifies that the operator accepts custom video parameters.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpWithCustomParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "custom_resource",
      holoscan::Arg("port") = uint16_t{48020});

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "custom_upstream_op",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 60u);

  ASSERT_NE(upstream_op_, nullptr);
  EXPECT_EQ(upstream_op_->name(), "custom_upstream_op");
}

/**
 * @brief Test upstream operator with various video resolutions
 *
 * Verifies that the operator accepts different video resolutions.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpVideoResolutionParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "resolution_resource");

  // Standard Definition
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "sd_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(upstream_op_, nullptr);

  // HD
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "hd_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(upstream_op_, nullptr);

  // Full HD
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "fhd_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(upstream_op_, nullptr);

  // 4K
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "4k_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 3840u,
      holoscan::Arg("height") = 2160u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(upstream_op_, nullptr);
}

/**
 * @brief Test upstream operator with various frame rates
 *
 * Verifies that the operator accepts different frame rates.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpFrameRateParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "fps_resource");

  // Low frame rate
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "low_fps_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 15u);
  ASSERT_NE(upstream_op_, nullptr);

  // Standard frame rate
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "standard_fps_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(upstream_op_, nullptr);

  // High frame rate
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "high_fps_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u);
  ASSERT_NE(upstream_op_, nullptr);

  // Very high frame rate
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "very_high_fps_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 120u);
  ASSERT_NE(upstream_op_, nullptr);
}

/**
 * @brief Test upstream operator setup method
 *
 * Verifies that the setup() method can be called without crashing.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpSetup) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "setup_resource");

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "test_setup_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(upstream_op_, nullptr);

  // Create an operator spec for setup testing
  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());

  // Test that setup doesn't crash (basic smoke test)
  EXPECT_NO_THROW(upstream_op_->setup(*spec));
}

/**
 * @brief Test upstream operator cleanup
 *
 * Verifies that the operator can be properly destroyed.
 */
TEST_F(StreamingServerUpstreamOpTest, UpstreamOpCleanup) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "cleanup_resource");

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "cleanup_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(upstream_op_, nullptr);

  // Test that cleanup doesn't crash
  EXPECT_NO_THROW({
    upstream_op_.reset();
  });

  EXPECT_EQ(upstream_op_, nullptr);
}

/**
 * @brief Test multiple upstream operator instances
 *
 * Verifies that multiple operators can share a resource.
 */
TEST_F(StreamingServerUpstreamOpTest, MultipleUpstreamOpInstances) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "shared_resource");

  auto upstream1 = fragment_->make_operator<StreamingServerUpstreamOp>(
      "upstream_1",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  auto upstream2 = fragment_->make_operator<StreamingServerUpstreamOp>(
      "upstream_2",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u);

  ASSERT_NE(upstream1, nullptr);
  ASSERT_NE(upstream2, nullptr);

  EXPECT_EQ(upstream1->name(), "upstream_1");
  EXPECT_EQ(upstream2->name(), "upstream_2");

  EXPECT_NE(upstream1, upstream2);
}

// ====================================================================================
// StreamingServerDownstreamOp Tests
// ====================================================================================

/**
 * @brief Test fixture for StreamingServerDownstreamOp unit tests
 *
 * This test fixture provides a Holoscan Fragment context for testing
 * the StreamingServerDownstreamOp operator.
 */
class StreamingServerDownstreamOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a minimal fragment for testing
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    downstream_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingServerDownstreamOp> downstream_op_;
};

/**
 * @brief Test basic downstream operator creation
 *
 * Verifies that the StreamingServerDownstreamOp can be instantiated with a
 * shared resource and standard parameters.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpBasicInitialization) {
  // Create shared resource
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "downstream_resource",
      holoscan::Arg("port") = uint16_t{48010});

  // Create downstream operator
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "test_downstream_op",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(downstream_op_, nullptr);
  EXPECT_EQ(downstream_op_->name(), "test_downstream_op");
  EXPECT_EQ(downstream_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

/**
 * @brief Test downstream operator with custom parameters
 *
 * Verifies that the operator accepts custom video parameters.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpWithCustomParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "custom_resource",
      holoscan::Arg("port") = uint16_t{48020});

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "custom_downstream_op",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 60u);

  ASSERT_NE(downstream_op_, nullptr);
  EXPECT_EQ(downstream_op_->name(), "custom_downstream_op");
}

/**
 * @brief Test downstream operator with various video resolutions
 *
 * Verifies that the operator accepts different video resolutions.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpVideoResolutionParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "resolution_resource");

  // Standard Definition
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "sd_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(downstream_op_, nullptr);

  // HD
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "hd_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(downstream_op_, nullptr);

  // Full HD
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "fhd_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(downstream_op_, nullptr);

  // 4K
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "4k_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 3840u,
      holoscan::Arg("height") = 2160u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(downstream_op_, nullptr);
}

/**
 * @brief Test downstream operator with various frame rates
 *
 * Verifies that the operator accepts different frame rates.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpFrameRateParameters) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "fps_resource");

  // Low frame rate
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "low_fps_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 15u);
  ASSERT_NE(downstream_op_, nullptr);

  // Standard frame rate
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "standard_fps_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);
  ASSERT_NE(downstream_op_, nullptr);

  // High frame rate
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "high_fps_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u);
  ASSERT_NE(downstream_op_, nullptr);

  // Very high frame rate
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "very_high_fps_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 120u);
  ASSERT_NE(downstream_op_, nullptr);
}

/**
 * @brief Test downstream operator setup method
 *
 * Verifies that the setup() method can be called without crashing.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpSetup) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "setup_resource");

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "test_setup_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(downstream_op_, nullptr);

  // Create an operator spec for setup testing
  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());

  // Test that setup doesn't crash (basic smoke test)
  EXPECT_NO_THROW(downstream_op_->setup(*spec));
}

/**
 * @brief Test downstream operator cleanup
 *
 * Verifies that the operator can be properly destroyed.
 */
TEST_F(StreamingServerDownstreamOpTest, DownstreamOpCleanup) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "cleanup_resource");

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "cleanup_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(downstream_op_, nullptr);

  // Test that cleanup doesn't crash
  EXPECT_NO_THROW({
    downstream_op_.reset();
  });

  EXPECT_EQ(downstream_op_, nullptr);
}

/**
 * @brief Test multiple downstream operator instances
 *
 * Verifies that multiple operators can share a resource.
 */
TEST_F(StreamingServerDownstreamOpTest, MultipleDownstreamOpInstances) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "shared_resource");

  auto downstream1 = fragment_->make_operator<StreamingServerDownstreamOp>(
      "downstream_1",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = 30u);

  auto downstream2 = fragment_->make_operator<StreamingServerDownstreamOp>(
      "downstream_2",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u);

  ASSERT_NE(downstream1, nullptr);
  ASSERT_NE(downstream2, nullptr);

  EXPECT_EQ(downstream1->name(), "downstream_1");
  EXPECT_EQ(downstream2->name(), "downstream_2");

  EXPECT_NE(downstream1, downstream2);
}

// ====================================================================================
// Integration Tests - Resource with Operators
// ====================================================================================

/**
 * @brief Test fixture for integration tests
 *
 * Tests the integration between StreamingServerResource and operators.
 */
class StreamingServerIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
};

/**
 * @brief Test resource integration with upstream operator
 *
 * Verifies that a resource can be properly shared with an upstream operator.
 */
TEST_F(StreamingServerIntegrationTest, ResourceWithUpstreamOp) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "integration_resource",
      holoscan::Arg("port") = uint16_t{48010});

  auto upstream_op = fragment_->make_operator<StreamingServerUpstreamOp>(
      "integration_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(resource, nullptr);
  ASSERT_NE(upstream_op, nullptr);
  EXPECT_EQ(upstream_op->name(), "integration_upstream");
}

/**
 * @brief Test resource integration with downstream operator
 *
 * Verifies that a resource can be properly shared with a downstream operator.
 */
TEST_F(StreamingServerIntegrationTest, ResourceWithDownstreamOp) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "integration_resource",
      holoscan::Arg("port") = uint16_t{48020});

  auto downstream_op = fragment_->make_operator<StreamingServerDownstreamOp>(
      "integration_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(resource, nullptr);
  ASSERT_NE(downstream_op, nullptr);
  EXPECT_EQ(downstream_op->name(), "integration_downstream");
}

/**
 * @brief Test resource integration with both upstream and downstream operators
 *
 * Verifies that a single resource can be shared between both operator types,
 * which is the typical production usage pattern.
 */
TEST_F(StreamingServerIntegrationTest, ResourceWithBothOps) {
  auto resource = fragment_->make_resource<StreamingServerResource>(
      "shared_resource",
      holoscan::Arg("port") = uint16_t{48030},
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = true);

  auto upstream_op = fragment_->make_operator<StreamingServerUpstreamOp>(
      "shared_upstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);

  auto downstream_op = fragment_->make_operator<StreamingServerDownstreamOp>(
      "shared_downstream",
      holoscan::Arg("video_streaming_server_resource") = resource,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 30u);

  ASSERT_NE(resource, nullptr);
  ASSERT_NE(upstream_op, nullptr);
  ASSERT_NE(downstream_op, nullptr);

  EXPECT_EQ(upstream_op->name(), "shared_upstream");
  EXPECT_EQ(downstream_op->name(), "shared_downstream");
}

}  // namespace holoscan::ops

