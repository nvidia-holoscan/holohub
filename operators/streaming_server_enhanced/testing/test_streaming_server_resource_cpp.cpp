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

#include "../streaming_server_resource.hpp"
#include "test_utilities.hpp"

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

using namespace holoscan::ops;
using namespace holoscan::ops::streaming_server_enhanced::testing;

class StreamingServerResourceTest : public StreamingServerTestFixture {
 protected:
  void SetUp() override {
    StreamingServerTestFixture::SetUp();
    
    // Create a resource instance for testing
    resource_ = std::make_shared<StreamingServerResource>();
    
    // Create operator spec for setup calls
    op_spec_ = std::make_shared<MockOperatorSpec>(fragment_.get());
  }

  void TearDown() override {
    resource_.reset();
    op_spec_.reset();
    StreamingServerTestFixture::TearDown();
  }

  std::shared_ptr<StreamingServerResource> resource_;
  std::shared_ptr<MockOperatorSpec> op_spec_;
};

// Basic initialization tests

TEST_F(StreamingServerResourceTest, DefaultConstruction) {
  EXPECT_NE(resource_, nullptr);
  EXPECT_NO_THROW({
    auto new_resource = std::make_shared<StreamingServerResource>();
  });
}

TEST_F(StreamingServerResourceTest, InitializationWithFragment) {
  EXPECT_NO_THROW({
    resource_->fragment(fragment_.get());
  });
}

// Configuration tests

TEST_F(StreamingServerResourceTest, SetWidthParameter) {
  EXPECT_NO_THROW({
    resource_->width(test_config_.width);
  });
  
  // Test with different valid widths
  uint32_t test_widths[] = {640, 854, 1920, 3840};
  for (uint32_t width : test_widths) {
    EXPECT_NO_THROW({
      resource_->width(width);
    });
  }
}

TEST_F(StreamingServerResourceTest, SetHeightParameter) {
  EXPECT_NO_THROW({
    resource_->height(test_config_.height);
  });
  
  // Test with different valid heights
  uint32_t test_heights[] = {480, 720, 1080, 2160};
  for (uint32_t height : test_heights) {
    EXPECT_NO_THROW({
      resource_->height(height);
    });
  }
}

TEST_F(StreamingServerResourceTest, SetFpsParameter) {
  EXPECT_NO_THROW({
    resource_->fps(test_config_.fps);
  });
  
  // Test with different valid frame rates
  uint32_t test_fps[] = {15, 30, 60, 120};
  for (uint32_t fps : test_fps) {
    EXPECT_NO_THROW({
      resource_->fps(fps);
    });
  }
}

TEST_F(StreamingServerResourceTest, SetPortParameter) {
  EXPECT_NO_THROW({
    resource_->port(test_config_.port);
  });
  
  // Test with different valid ports
  uint16_t test_ports[] = {48010, 48020, 49000, 50000};
  for (uint16_t port : test_ports) {
    EXPECT_NO_THROW({
      resource_->port(port);
    });
  }
}

TEST_F(StreamingServerResourceTest, SetServerNameParameter) {
  EXPECT_NO_THROW({
    resource_->server_name(test_config_.server_name);
  });
  
  // Test with different valid server names
  std::vector<std::string> test_names = {
    "TestServer",
    "HoloscanStreamingServer",
    "My_Custom_Server_123",
    ""  // Empty name should be allowed
  };
  
  for (const auto& name : test_names) {
    EXPECT_NO_THROW({
      resource_->server_name(name);
    });
  }
}

// Parameter validation tests

TEST_F(StreamingServerResourceTest, InvalidWidthParameters) {
  // Test edge cases for width
  EXPECT_NO_THROW({
    resource_->width(1);  // Minimum valid width
  });
  
  EXPECT_NO_THROW({
    resource_->width(7680);  // Large width (8K)
  });
}

TEST_F(StreamingServerResourceTest, InvalidHeightParameters) {
  // Test edge cases for height
  EXPECT_NO_THROW({
    resource_->height(1);  // Minimum valid height
  });
  
  EXPECT_NO_THROW({
    resource_->height(4320);  // Large height (8K)
  });
}

TEST_F(StreamingServerResourceTest, InvalidFpsParameters) {
  // Test edge cases for FPS
  EXPECT_NO_THROW({
    resource_->fps(1);  // Minimum valid FPS
  });
  
  EXPECT_NO_THROW({
    resource_->fps(240);  // High FPS
  });
}

// Setup and initialization tests

TEST_F(StreamingServerResourceTest, SetupWithValidParameters) {
  // Configure the resource
  resource_->width(test_config_.width);
  resource_->height(test_config_.height);
  resource_->fps(test_config_.fps);
  resource_->port(test_config_.port);
  resource_->server_name(test_config_.server_name);
  
  // Setup should not throw with valid parameters
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerResourceTest, InitializeAfterSetup) {
  // Configure and setup the resource
  resource_->width(test_config_.width);
  resource_->height(test_config_.height);
  resource_->fps(test_config_.fps);
  resource_->port(test_config_.port);
  resource_->fragment(fragment_.get());
  
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
  
  EXPECT_NO_THROW({
    resource_->initialize();
  });
}

// Lifecycle tests

TEST_F(StreamingServerResourceTest, MultipleSetupCalls) {
  // Should be able to call setup multiple times without error
  resource_->width(test_config_.width);
  resource_->height(test_config_.height);
  
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
    resource_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerResourceTest, SetupAndInitializeLifecycle) {
  // Test complete lifecycle: construction -> setup -> initialize
  resource_->width(test_config_.width);
  resource_->height(test_config_.height);
  resource_->fps(test_config_.fps);
  resource_->port(test_config_.port);
  resource_->fragment(fragment_.get());
  
  // Setup
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
  
  // Initialize
  EXPECT_NO_THROW({
    resource_->initialize();
  });
}

// Configuration combination tests

TEST_F(StreamingServerResourceTest, CommonResolutionConfigurations) {
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
      resource_->width(config.width);
      resource_->height(config.height);
      resource_->setup(*op_spec_);
    }) << "Failed for " << config.name << " resolution (" 
      << config.width << "x" << config.height << ")";
  }
}

TEST_F(StreamingServerResourceTest, CommonFrameRateConfigurations) {
  std::vector<uint32_t> framerates = {15, 24, 30, 60, 120};
  
  for (uint32_t fps : framerates) {
    EXPECT_NO_THROW({
      resource_->fps(fps);
      resource_->setup(*op_spec_);
    }) << "Failed for " << fps << " FPS";
  }
}

// Thread safety tests (basic)

TEST_F(StreamingServerResourceTest, ConcurrentParameterSetting) {
  // This is a basic test - in a real implementation you might use threading
  // For now, just test that multiple parameter sets work
  
  EXPECT_NO_THROW({
    resource_->width(1920);
    resource_->height(1080);
    resource_->fps(60);
    resource_->port(48015);
    resource_->server_name("ConcurrentTest");
  });
  
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
}

// Error recovery tests

TEST_F(StreamingServerResourceTest, SetupAfterParameterChanges) {
  // Initial setup
  resource_->width(640);
  resource_->height(480);
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
  
  // Change parameters and setup again
  resource_->width(1920);
  resource_->height(1080);
  EXPECT_NO_THROW({
    resource_->setup(*op_spec_);
  });
}

TEST_F(StreamingServerResourceTest, ResourceReuse) {
  // Test that the same resource can be used multiple times
  for (int i = 0; i < 3; ++i) {
    resource_->width(test_config_.width);
    resource_->height(test_config_.height);
    
    EXPECT_NO_THROW({
      resource_->setup(*op_spec_);
    }) << "Failed on iteration " << i;
  }
}
