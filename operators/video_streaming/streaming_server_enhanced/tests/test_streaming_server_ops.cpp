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
#include "../streaming_server_resource.hpp"
#include "../streaming_server_upstream_op.hpp"
#include "../streaming_server_downstream_op.hpp"

namespace holoscan::ops {

/**
 * @brief Test fixture for StreamingServerResource unit tests
 * 
 * This test fixture provides a Holoscan Fragment context for testing
 * the StreamingServerResource in isolation.
 */
class StreamingServerResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
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
 * @brief Test fixture for StreamingServerUpstreamOp unit tests
 */
class StreamingServerUpstreamOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    upstream_op_.reset();
    resource_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingServerResource> resource_;
  std::shared_ptr<StreamingServerUpstreamOp> upstream_op_;
};

/**
 * @brief Test fixture for StreamingServerDownstreamOp unit tests
 */
class StreamingServerDownstreamOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    downstream_op_.reset();
    resource_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<StreamingServerResource> resource_;
  std::shared_ptr<StreamingServerDownstreamOp> downstream_op_;
};

// ====================================================================================
// StreamingServerResource Tests
// ====================================================================================

/**
 * @brief Test basic resource creation
 * 
 * Verifies that the StreamingServerResource can be created with default parameters.
 */
TEST_F(StreamingServerResourceTest, BasicInitialization) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_server_resource"
  );

  ASSERT_NE(resource_, nullptr);
  EXPECT_EQ(resource_->name(), "test_server_resource");
}

/**
 * @brief Test resource creation with custom configuration
 * 
 * Verifies that the resource accepts custom port, resolution, and FPS parameters.
 */
TEST_F(StreamingServerResourceTest, CustomConfiguration) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "custom_server_resource",
      holoscan::Arg("port") = uint16_t{8080},
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = uint16_t{60}
  );

  ASSERT_NE(resource_, nullptr);
  EXPECT_EQ(resource_->name(), "custom_server_resource");
}

/**
 * @brief Test resource with upstream/downstream configuration
 * 
 * Verifies that the resource can be configured to enable/disable
 * upstream and downstream streaming.
 */
TEST_F(StreamingServerResourceTest, StreamingDirectionConfiguration) {
  // Upstream only
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "upstream_only_resource",
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = false
  );
  ASSERT_NE(resource_, nullptr);

  // Downstream only
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "downstream_only_resource",
      holoscan::Arg("enable_upstream") = false,
      holoscan::Arg("enable_downstream") = true
  );
  ASSERT_NE(resource_, nullptr);

  // Both directions (default)
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "bidirectional_resource",
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = true
  );
  ASSERT_NE(resource_, nullptr);

  // Both disabled
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "disabled_resource",
      holoscan::Arg("enable_upstream") = false,
      holoscan::Arg("enable_downstream") = false
  );
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource with multi-instance configuration
 * 
 * Verifies that the resource can be configured for multi-instance mode.
 */
TEST_F(StreamingServerResourceTest, MultiInstanceConfiguration) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "multi_instance_resource",
      holoscan::Arg("is_multi_instance") = true,
      holoscan::Arg("server_name") = std::string("MultiInstanceServer")
  );

  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource with different resolutions
 * 
 * Verifies that the resource accepts various video resolutions.
 */
TEST_F(StreamingServerResourceTest, VariousResolutions) {
  // SD
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "sd_resource",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u
  );
  ASSERT_NE(resource_, nullptr);

  // HD
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "hd_resource",
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u
  );
  ASSERT_NE(resource_, nullptr);

  // Full HD
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "fhd_resource",
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u
  );
  ASSERT_NE(resource_, nullptr);

  // 4K
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "4k_resource",
      holoscan::Arg("width") = 3840u,
      holoscan::Arg("height") = 2160u
  );
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource with different frame rates
 * 
 * Verifies that the resource accepts various frame rates.
 */
TEST_F(StreamingServerResourceTest, VariousFrameRates) {
  // Low FPS
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "low_fps_resource",
      holoscan::Arg("fps") = uint16_t{15}
  );
  ASSERT_NE(resource_, nullptr);

  // Standard FPS
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "standard_fps_resource",
      holoscan::Arg("fps") = uint16_t{30}
  );
  ASSERT_NE(resource_, nullptr);

  // High FPS
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "high_fps_resource",
      holoscan::Arg("fps") = uint16_t{60}
  );
  ASSERT_NE(resource_, nullptr);

  // Very high FPS
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "very_high_fps_resource",
      holoscan::Arg("fps") = uint16_t{120}
  );
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource with different port numbers
 * 
 * Verifies that the resource accepts various port configurations.
 */
TEST_F(StreamingServerResourceTest, VariousPortNumbers) {
  // Default port
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "default_port_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  // Custom port
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "custom_port_resource",
      holoscan::Arg("port") = uint16_t{8080}
  );
  ASSERT_NE(resource_, nullptr);

  // High port
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "high_port_resource",
      holoscan::Arg("port") = uint16_t{65535}
  );
  ASSERT_NE(resource_, nullptr);
}

/**
 * @brief Test resource cleanup
 * 
 * Verifies that the resource can be properly destroyed without crashing.
 */
TEST_F(StreamingServerResourceTest, ResourceCleanup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "cleanup_test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );

  ASSERT_NE(resource_, nullptr);
  
  EXPECT_NO_THROW({
    resource_.reset();
  });
  
  EXPECT_EQ(resource_, nullptr);
}

// ====================================================================================
// StreamingServerUpstreamOp Tests
// ====================================================================================

/**
 * @brief Test basic upstream operator creation
 * 
 * Verifies that the StreamingServerUpstreamOp can be created with a
 * StreamingServerResource.
 */
TEST_F(StreamingServerUpstreamOpTest, BasicInitialization) {
  // Create resource first
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  // Create upstream operator
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "test_upstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  ASSERT_NE(upstream_op_, nullptr);
  EXPECT_EQ(upstream_op_->name(), "test_upstream_op");
  EXPECT_EQ(upstream_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

/**
 * @brief Test upstream operator with custom video parameters
 * 
 * Verifies that the upstream operator can override the resource's
 * default video parameters.
 */
TEST_F(StreamingServerUpstreamOpTest, CustomVideoParameters) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u
  );
  ASSERT_NE(resource_, nullptr);

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "custom_params_upstream_op",
      holoscan::Arg("streaming_server_resource") = resource_,
      holoscan::Arg("width") = 1280u,
      holoscan::Arg("height") = 720u,
      holoscan::Arg("fps") = 60u
  );

  ASSERT_NE(upstream_op_, nullptr);
  EXPECT_EQ(upstream_op_->name(), "custom_params_upstream_op");
}

/**
 * @brief Test upstream operator setup
 * 
 * Verifies that the setup() method can be called without crashing.
 */
TEST_F(StreamingServerUpstreamOpTest, OperatorSetup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "setup_test_upstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(upstream_op_, nullptr);

  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(upstream_op_->setup(*spec));
}

/**
 * @brief Test upstream operator cleanup
 * 
 * Verifies that the upstream operator can be properly destroyed.
 */
TEST_F(StreamingServerUpstreamOpTest, OperatorCleanup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "cleanup_test_upstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(upstream_op_, nullptr);

  EXPECT_NO_THROW({
    upstream_op_.reset();
  });
  
  EXPECT_EQ(upstream_op_, nullptr);
}

// ====================================================================================
// StreamingServerDownstreamOp Tests
// ====================================================================================

/**
 * @brief Test basic downstream operator creation
 * 
 * Verifies that the StreamingServerDownstreamOp can be created with a
 * StreamingServerResource.
 */
TEST_F(StreamingServerDownstreamOpTest, BasicInitialization) {
  // Create resource first
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  // Create downstream operator
  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "test_downstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  ASSERT_NE(downstream_op_, nullptr);
  EXPECT_EQ(downstream_op_->name(), "test_downstream_op");
  EXPECT_EQ(downstream_op_->operator_type(), holoscan::Operator::OperatorType::kNative);
}

/**
 * @brief Test downstream operator with custom video parameters
 * 
 * Verifies that the downstream operator can override the resource's
 * default video parameters.
 */
TEST_F(StreamingServerDownstreamOpTest, CustomVideoParameters) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u
  );
  ASSERT_NE(resource_, nullptr);

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "custom_params_downstream_op",
      holoscan::Arg("streaming_server_resource") = resource_,
      holoscan::Arg("width") = 1920u,
      holoscan::Arg("height") = 1080u,
      holoscan::Arg("fps") = 30u
  );

  ASSERT_NE(downstream_op_, nullptr);
  EXPECT_EQ(downstream_op_->name(), "custom_params_downstream_op");
}

/**
 * @brief Test downstream operator setup
 * 
 * Verifies that the setup() method can be called without crashing.
 */
TEST_F(StreamingServerDownstreamOpTest, OperatorSetup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "setup_test_downstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(downstream_op_, nullptr);

  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(downstream_op_->setup(*spec));
}

/**
 * @brief Test downstream operator cleanup
 * 
 * Verifies that the downstream operator can be properly destroyed.
 */
TEST_F(StreamingServerDownstreamOpTest, OperatorCleanup) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  downstream_op_ = fragment_->make_operator<StreamingServerDownstreamOp>(
      "cleanup_test_downstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(downstream_op_, nullptr);

  EXPECT_NO_THROW({
    downstream_op_.reset();
  });
  
  EXPECT_EQ(downstream_op_, nullptr);
}

// ====================================================================================
// Integrated Server Tests (Resource + Both Operators)
// ====================================================================================

/**
 * @brief Test complete server setup with both upstream and downstream operators
 * 
 * Verifies that a resource can be shared between upstream and downstream operators.
 */
TEST_F(StreamingServerUpstreamOpTest, SharedResourceConfiguration) {
  // Create shared resource
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "shared_resource",
      holoscan::Arg("port") = uint16_t{48010},
      holoscan::Arg("width") = 854u,
      holoscan::Arg("height") = 480u,
      holoscan::Arg("fps") = uint16_t{30},
      holoscan::Arg("enable_upstream") = true,
      holoscan::Arg("enable_downstream") = true
  );
  ASSERT_NE(resource_, nullptr);

  // Create upstream operator with shared resource
  auto upstream = fragment_->make_operator<StreamingServerUpstreamOp>(
      "upstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(upstream, nullptr);

  // Create downstream operator with shared resource
  auto downstream = fragment_->make_operator<StreamingServerDownstreamOp>(
      "downstream_op",
      holoscan::Arg("streaming_server_resource") = resource_
  );
  ASSERT_NE(downstream, nullptr);

  // Verify both operators were created successfully
  EXPECT_EQ(upstream->name(), "upstream_op");
  EXPECT_EQ(downstream->name(), "downstream_op");
}

/**
 * @brief Test multiple operator instances with same resource
 * 
 * Verifies that multiple operators can share the same resource
 * without conflicts.
 */
TEST_F(StreamingServerUpstreamOpTest, MultipleOperatorsSharedResource) {
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "shared_resource",
      holoscan::Arg("port") = uint16_t{48010}
  );
  ASSERT_NE(resource_, nullptr);

  auto upstream1 = fragment_->make_operator<StreamingServerUpstreamOp>(
      "upstream_op_1",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  auto upstream2 = fragment_->make_operator<StreamingServerUpstreamOp>(
      "upstream_op_2",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  auto downstream1 = fragment_->make_operator<StreamingServerDownstreamOp>(
      "downstream_op_1",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  // Verify all operators were created successfully
  ASSERT_NE(upstream1, nullptr);
  ASSERT_NE(upstream2, nullptr);
  ASSERT_NE(downstream1, nullptr);

  // Verify different upstream instances are distinct
  EXPECT_NE(upstream1, upstream2);
  
  // All operators share the same resource and are distinct instances
  // (We cannot directly compare different operator types with EXPECT_NE)
}

}  // namespace holoscan::ops

