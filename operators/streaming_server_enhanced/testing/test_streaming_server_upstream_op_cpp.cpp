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
#include <gmock/gmock.h>

#include "../streaming_server_upstream_op.hpp"
#include "../streaming_server_resource.hpp"
#include "test_utilities.hpp"

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

using namespace holoscan::ops;
using namespace holoscan::ops::streaming_server_enhanced::testing;

class StreamingServerUpstreamOpTest : public StreamingServerTestFixture {
 protected:
  void SetUp() override {
    StreamingServerTestFixture::SetUp();
    
    // Create resource for the operator
    resource_ = std::make_shared<StreamingServerResource>();
    resource_->width(test_config_.width);
    resource_->height(test_config_.height);
    resource_->fps(test_config_.fps);
    resource_->port(test_config_.port);
    resource_->fragment(fragment_.get());
    
    // Create operator instance
    upstream_op_ = std::make_shared<StreamingServerUpstreamOp>();
    upstream_op_->fragment(fragment_.get());
    
    // Create operator spec for setup calls
    op_spec_ = std::make_shared<MockOperatorSpec>(fragment_.get());
    
    // Set up input/output contexts for the operator
    input_context_ = std::make_shared<MockInputContext>(execution_context_.get(), 
                                                         upstream_op_.get());
    output_context_ = std::make_shared<MockOutputContext>(execution_context_.get(), 
                                                           upstream_op_.get());
  }

  void TearDown() override {
    upstream_op_.reset();
    resource_.reset();
    op_spec_.reset();
    StreamingServerTestFixture::TearDown();
  }

  std::shared_ptr<StreamingServerUpstreamOp> upstream_op_;
  std::shared_ptr<StreamingServerResource> resource_;
  std::shared_ptr<MockOperatorSpec> op_spec_;
};

// Basic initialization tests

TEST_F(StreamingServerUpstreamOpTest, DefaultConstruction) {
  EXPECT_NE(upstream_op_, nullptr);
  EXPECT_NO_THROW({
    auto new_op = std::make_shared<StreamingServerUpstreamOp>();
  });
}

TEST_F(StreamingServerUpstreamOpTest, InitializationWithFragment) {
  EXPECT_NO_THROW({
    upstream_op_->fragment(fragment_.get());
  });
}

// Parameter configuration tests

TEST_F(StreamingServerUpstreamOpTest, SetWidthParameter) {
  EXPECT_NO_THROW({
    upstream_op_->width(test_config_.width);
  });
  
  // Test with different valid widths
  uint32_t test_widths[] = {640, 854, 1920, 3840};
  for (uint32_t width : test_widths) {
    EXPECT_NO_THROW({
      upstream_op_->width(width);
    });
  }
}

TEST_F(StreamingServerUpstreamOpTest, SetHeightParameter) {
  EXPECT_NO_THROW({
    upstream_op_->height(test_config_.height);
  });
  
  // Test with different valid heights
  uint32_t test_heights[] = {480, 720, 1080, 2160};
  for (uint32_t height : test_heights) {
    EXPECT_NO_THROW({
      upstream_op_->height(height);
    });
  }
}

TEST_F(StreamingServerUpstreamOpTest, SetFpsParameter) {
  EXPECT_NO_THROW({
    upstream_op_->fps(test_config_.fps);
  });
  
  // Test with different valid frame rates
  uint32_t test_fps[] = {15, 30, 60, 120};
  for (uint32_t fps : test_fps) {
    EXPECT_NO_THROW({
      upstream_op_->fps(fps);
    });
  }
}

TEST_F(StreamingServerUpstreamOpTest, SetStreamingServerResource) {
  EXPECT_NO_THROW({
    upstream_op_->streaming_server_resource(resource_);
  });
}

// Setup tests

TEST_F(StreamingServerUpstreamOpTest, SetupWithValidParameters) {
  // Configure the operator
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  upstream_op_->streaming_server_resource(resource_);
  
  // Setup should not throw with valid parameters
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, SetupWithoutResource) {
  // Configure the operator without resource
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  
  // Setup might handle missing resource gracefully
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, InitializeAfterSetup) {
  // Configure and setup the operator
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  upstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
  
  EXPECT_NO_THROW({
    upstream_op_->initialize();
  });
}

// Lifecycle tests

TEST_F(StreamingServerUpstreamOpTest, CompleteLifecycle) {
  // Test complete operator lifecycle: construction -> setup -> initialize
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  upstream_op_->streaming_server_resource(resource_);
  
  // Setup
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
  
  // Initialize
  EXPECT_NO_THROW({
    upstream_op_->initialize();
  });
}

// Parameter validation tests

TEST_F(StreamingServerUpstreamOpTest, ValidResolutionCombinations) {
  struct ResolutionConfig {
    uint32_t width;
    uint32_t height;
    std::string name;
  };
  
  std::vector<ResolutionConfig> configs = {
    {640, 480, "VGA"},
    {854, 480, "FWVGA"},
    {1280, 720, "HD"},
    {1920, 1080, "Full HD"},
    {3840, 2160, "4K UHD"}
  };
  
  for (const auto& config : configs) {
    EXPECT_NO_THROW({
      upstream_op_->width(config.width);
      upstream_op_->height(config.height);
      upstream_op_->streaming_server_resource(resource_);
      upstream_op_->setup(*op_spec_);
    }) << "Failed for " << config.name << " resolution (" 
      << config.width << "x" << config.height << ")";
  }
}

TEST_F(StreamingServerUpstreamOpTest, ValidFrameRates) {
  std::vector<uint32_t> framerates = {15, 24, 30, 60, 120};
  
  for (uint32_t fps : framerates) {
    EXPECT_NO_THROW({
      upstream_op_->width(test_config_.width);
      upstream_op_->height(test_config_.height);
      upstream_op_->fps(fps);
      upstream_op_->streaming_server_resource(resource_);
      upstream_op_->setup(*op_spec_);
    }) << "Failed for " << fps << " FPS";
  }
}

// Edge case tests

TEST_F(StreamingServerUpstreamOpTest, MinimumValidParameters) {
  EXPECT_NO_THROW({
    upstream_op_->width(1);
    upstream_op_->height(1);
    upstream_op_->fps(1);
    upstream_op_->streaming_server_resource(resource_);
    upstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, LargeResolutionParameters) {
  EXPECT_NO_THROW({
    upstream_op_->width(7680);  // 8K width
    upstream_op_->height(4320); // 8K height
    upstream_op_->fps(30);
    upstream_op_->streaming_server_resource(resource_);
    upstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, HighFrameRate) {
  EXPECT_NO_THROW({
    upstream_op_->width(test_config_.width);
    upstream_op_->height(test_config_.height);
    upstream_op_->fps(240);  // High frame rate
    upstream_op_->streaming_server_resource(resource_);
    upstream_op_->setup(*op_spec_);
  });
}

// Multiple setup tests

TEST_F(StreamingServerUpstreamOpTest, MultipleSetupCalls) {
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  upstream_op_->streaming_server_resource(resource_);
  
  // Should be able to call setup multiple times
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
    upstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, ParameterChangeAfterSetup) {
  // Initial setup
  upstream_op_->width(640);
  upstream_op_->height(480);
  upstream_op_->fps(30);
  upstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
  
  // Change parameters and setup again
  upstream_op_->width(1920);
  upstream_op_->height(1080);
  upstream_op_->fps(60);
  
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
}

// Resource management tests

TEST_F(StreamingServerUpstreamOpTest, ResourceSharing) {
  // Create another operator that shares the same resource
  auto upstream_op2 = std::make_shared<StreamingServerUpstreamOp>();
  upstream_op2->fragment(fragment_.get());
  
  // Both operators should be able to use the same resource
  EXPECT_NO_THROW({
    upstream_op_->streaming_server_resource(resource_);
    upstream_op2->streaming_server_resource(resource_);
    
    upstream_op_->width(test_config_.width);
    upstream_op_->height(test_config_.height);
    upstream_op_->setup(*op_spec_);
    
    upstream_op2->width(test_config_.width);
    upstream_op2->height(test_config_.height);
    upstream_op2->setup(*op_spec_);
  });
}

TEST_F(StreamingServerUpstreamOpTest, ResourceParameterConsistency) {
  // Resource and operator parameters should be compatible
  resource_->width(1920);
  resource_->height(1080);
  resource_->fps(60);
  
  upstream_op_->width(1920);
  upstream_op_->height(1080);
  upstream_op_->fps(60);
  upstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
  });
}

// Compute method tests (basic structure)

TEST_F(StreamingServerUpstreamOpTest, ComputeMethodExists) {
  // Set up the operator properly
  upstream_op_->width(test_config_.width);
  upstream_op_->height(test_config_.height);
  upstream_op_->fps(test_config_.fps);
  upstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    upstream_op_->setup(*op_spec_);
    upstream_op_->initialize();
  });
  
  // Note: We can't easily test the compute method without actual input/output contexts
  // and proper tensor infrastructure. This test just verifies the operator can be set up
  // for compute operations.
}
