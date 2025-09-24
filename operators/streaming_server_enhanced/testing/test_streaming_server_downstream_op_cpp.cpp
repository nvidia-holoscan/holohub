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

#include "../streaming_server_downstream_op.hpp"
#include "../streaming_server_resource.hpp"
#include "test_utilities.hpp"

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

using namespace holoscan::ops;
using namespace holoscan::ops::streaming_server_enhanced::testing;

class StreamingServerDownstreamOpTest : public StreamingServerTestFixture {
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
    downstream_op_ = std::make_shared<StreamingServerDownstreamOp>();
    downstream_op_->fragment(fragment_.get());
    
    // Create operator spec for setup calls
    op_spec_ = std::make_shared<MockOperatorSpec>(fragment_.get());
    
    // Set up input/output contexts for the operator
    input_context_ = std::make_shared<MockInputContext>(execution_context_.get(), 
                                                         downstream_op_.get());
    output_context_ = std::make_shared<MockOutputContext>(execution_context_.get(), 
                                                           downstream_op_.get());
  }

  void TearDown() override {
    downstream_op_.reset();
    resource_.reset();
    op_spec_.reset();
    StreamingServerTestFixture::TearDown();
  }

  std::shared_ptr<StreamingServerDownstreamOp> downstream_op_;
  std::shared_ptr<StreamingServerResource> resource_;
  std::shared_ptr<MockOperatorSpec> op_spec_;
};

// Basic initialization tests

TEST_F(StreamingServerDownstreamOpTest, DefaultConstruction) {
  EXPECT_NE(downstream_op_, nullptr);
  EXPECT_NO_THROW({
    auto new_op = std::make_shared<StreamingServerDownstreamOp>();
  });
}

TEST_F(StreamingServerDownstreamOpTest, InitializationWithFragment) {
  EXPECT_NO_THROW({
    downstream_op_->fragment(fragment_.get());
  });
}

// Parameter configuration tests

TEST_F(StreamingServerDownstreamOpTest, SetWidthParameter) {
  EXPECT_NO_THROW({
    downstream_op_->width(test_config_.width);
  });
  
  // Test with different valid widths
  uint32_t test_widths[] = {640, 854, 1920, 3840};
  for (uint32_t width : test_widths) {
    EXPECT_NO_THROW({
      downstream_op_->width(width);
    });
  }
}

TEST_F(StreamingServerDownstreamOpTest, SetHeightParameter) {
  EXPECT_NO_THROW({
    downstream_op_->height(test_config_.height);
  });
  
  // Test with different valid heights
  uint32_t test_heights[] = {480, 720, 1080, 2160};
  for (uint32_t height : test_heights) {
    EXPECT_NO_THROW({
      downstream_op_->height(height);
    });
  }
}

TEST_F(StreamingServerDownstreamOpTest, SetFpsParameter) {
  EXPECT_NO_THROW({
    downstream_op_->fps(test_config_.fps);
  });
  
  // Test with different valid frame rates
  uint32_t test_fps[] = {15, 30, 60, 120};
  for (uint32_t fps : test_fps) {
    EXPECT_NO_THROW({
      downstream_op_->fps(fps);
    });
  }
}

TEST_F(StreamingServerDownstreamOpTest, SetStreamingServerResource) {
  EXPECT_NO_THROW({
    downstream_op_->streaming_server_resource(resource_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, SetMirrorParameter) {
  EXPECT_NO_THROW({
    downstream_op_->mirror(true);
    downstream_op_->mirror(false);
  });
}

// Setup tests

TEST_F(StreamingServerDownstreamOpTest, SetupWithValidParameters) {
  // Configure the operator
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  downstream_op_->streaming_server_resource(resource_);
  
  // Setup should not throw with valid parameters
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, SetupWithMirrorEnabled) {
  // Configure the operator with mirroring
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  downstream_op_->mirror(true);
  downstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, SetupWithoutResource) {
  // Configure the operator without resource
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  
  // Setup might handle missing resource gracefully
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, InitializeAfterSetup) {
  // Configure and setup the operator
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  downstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
  
  EXPECT_NO_THROW({
    downstream_op_->initialize();
  });
}

// Lifecycle tests

TEST_F(StreamingServerDownstreamOpTest, CompleteLifecycle) {
  // Test complete operator lifecycle: construction -> setup -> initialize
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  downstream_op_->streaming_server_resource(resource_);
  
  // Setup
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
  
  // Initialize
  EXPECT_NO_THROW({
    downstream_op_->initialize();
  });
}

// Parameter validation tests

TEST_F(StreamingServerDownstreamOpTest, ValidResolutionCombinations) {
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
      downstream_op_->width(config.width);
      downstream_op_->height(config.height);
      downstream_op_->streaming_server_resource(resource_);
      downstream_op_->setup(*op_spec_);
    }) << "Failed for " << config.name << " resolution (" 
      << config.width << "x" << config.height << ")";
  }
}

TEST_F(StreamingServerDownstreamOpTest, ValidFrameRates) {
  std::vector<uint32_t> framerates = {15, 24, 30, 60, 120};
  
  for (uint32_t fps : framerates) {
    EXPECT_NO_THROW({
      downstream_op_->width(test_config_.width);
      downstream_op_->height(test_config_.height);
      downstream_op_->fps(fps);
      downstream_op_->streaming_server_resource(resource_);
      downstream_op_->setup(*op_spec_);
    }) << "Failed for " << fps << " FPS";
  }
}

// Mirror functionality tests

TEST_F(StreamingServerDownstreamOpTest, MirrorParameterCombinations) {
  std::vector<bool> mirror_settings = {true, false};
  
  for (bool mirror : mirror_settings) {
    EXPECT_NO_THROW({
      downstream_op_->width(test_config_.width);
      downstream_op_->height(test_config_.height);
      downstream_op_->fps(test_config_.fps);
      downstream_op_->mirror(mirror);
      downstream_op_->streaming_server_resource(resource_);
      downstream_op_->setup(*op_spec_);
    }) << "Failed with mirror = " << (mirror ? "true" : "false");
  }
}

TEST_F(StreamingServerDownstreamOpTest, MirrorWithDifferentResolutions) {
  struct ResolutionConfig {
    uint32_t width;
    uint32_t height;
    std::string name;
  };
  
  std::vector<ResolutionConfig> configs = {
    {640, 480, "VGA"},
    {1920, 1080, "Full HD"},
    {3840, 2160, "4K UHD"}
  };
  
  for (const auto& config : configs) {
    EXPECT_NO_THROW({
      downstream_op_->width(config.width);
      downstream_op_->height(config.height);
      downstream_op_->mirror(true);
      downstream_op_->streaming_server_resource(resource_);
      downstream_op_->setup(*op_spec_);
    }) << "Failed mirror test for " << config.name << " resolution";
  }
}

// Edge case tests

TEST_F(StreamingServerDownstreamOpTest, MinimumValidParameters) {
  EXPECT_NO_THROW({
    downstream_op_->width(1);
    downstream_op_->height(1);
    downstream_op_->fps(1);
    downstream_op_->streaming_server_resource(resource_);
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, LargeResolutionParameters) {
  EXPECT_NO_THROW({
    downstream_op_->width(7680);  // 8K width
    downstream_op_->height(4320); // 8K height
    downstream_op_->fps(30);
    downstream_op_->streaming_server_resource(resource_);
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, HighFrameRate) {
  EXPECT_NO_THROW({
    downstream_op_->width(test_config_.width);
    downstream_op_->height(test_config_.height);
    downstream_op_->fps(240);  // High frame rate
    downstream_op_->streaming_server_resource(resource_);
    downstream_op_->setup(*op_spec_);
  });
}

// Multiple setup tests

TEST_F(StreamingServerDownstreamOpTest, MultipleSetupCalls) {
  downstream_op_->width(test_config_.width);
  downstream_op_->height(test_config_.height);
  downstream_op_->fps(test_config_.fps);
  downstream_op_->streaming_server_resource(resource_);
  
  // Should be able to call setup multiple times
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
    downstream_op_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, ParameterChangeAfterSetup) {
  // Initial setup
  downstream_op_->width(640);
  downstream_op_->height(480);
  downstream_op_->fps(30);
  downstream_op_->mirror(false);
  downstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
  
  // Change parameters and setup again
  downstream_op_->width(1920);
  downstream_op_->height(1080);
  downstream_op_->fps(60);
  downstream_op_->mirror(true);
  
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
}

// Resource management tests

TEST_F(StreamingServerDownstreamOpTest, ResourceSharing) {
  // Create another operator that shares the same resource
  auto downstream_op2 = std::make_shared<StreamingServerDownstreamOp>();
  downstream_op2->fragment(fragment_.get());
  
  // Both operators should be able to use the same resource
  EXPECT_NO_THROW({
    downstream_op_->streaming_server_resource(resource_);
    downstream_op2->streaming_server_resource(resource_);
    
    downstream_op_->width(test_config_.width);
    downstream_op_->height(test_config_.height);
    downstream_op_->setup(*op_spec_);
    
    downstream_op2->width(test_config_.width);
    downstream_op2->height(test_config_.height);
    downstream_op2->setup(*op_spec_);
  });
}

TEST_F(StreamingServerDownstreamOpTest, ResourceParameterConsistency) {
  // Resource and operator parameters should be compatible
  resource_->width(1920);
  resource_->height(1080);
  resource_->fps(60);
  
  downstream_op_->width(1920);
  downstream_op_->height(1080);
  downstream_op_->fps(60);
  downstream_op_->streaming_server_resource(resource_);
  
  EXPECT_NO_THROW({
    downstream_op_->setup(*op_spec_);
  });
}

// Processing configuration tests

TEST_F(StreamingServerDownstreamOpTest, ProcessingPipelineConfiguration) {
  // Test different processing configurations
  struct ProcessingConfig {
    uint32_t width;
    uint32_t height;
    bool mirror;
    std::string description;
  };
  
  std::vector<ProcessingConfig> configs = {
    {854, 480, false, "Standard processing"},
    {854, 480, true, "Mirrored processing"},
    {1920, 1080, false, "HD processing"},
    {1920, 1080, true, "HD mirrored processing"}
  };
  
  for (const auto& config : configs) {
    EXPECT_NO_THROW({
      downstream_op_->width(config.width);
      downstream_op_->height(config.height);
      downstream_op_->mirror(config.mirror);
      downstream_op_->streaming_server_resource(resource_);
      downstream_op_->setup(*op_spec_);
    }) << "Failed for " << config.description;
  }
}
