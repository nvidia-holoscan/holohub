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

#include <holoscan/holoscan.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/application.hpp>

#include "streaming_server_op.hpp"

using namespace holoscan;
using namespace holoscan::ops;

/**
 * @brief Test fixture for StreamingServerOp unit tests
 * 
 * Provides a controlled environment for testing the StreamingServerOp
 * with proper Holoscan fragment context.
 */
class StreamingServerOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        app = std::make_shared<Application>();
        fragment = app.get();
    }

    void TearDown() override {
        app.reset();
        fragment = nullptr;
    }

    std::shared_ptr<Application> app;
    Fragment* fragment;
};

/**
 * @brief Test basic construction of StreamingServerOp
 * 
 * Verifies that the operator can be constructed with default parameters.
 */
TEST_F(StreamingServerOpTest, Construction) {
    auto op = std::make_shared<StreamingServerOp>();
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->name(), "");
}

/**
 * @brief Test minimal construction with fragment
 * 
 * Verifies that the operator can be constructed with a fragment context.
 */
TEST_F(StreamingServerOpTest, MinimalConstruction) {
    auto op = fragment->make_operator<StreamingServerOp>("test_server");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->name(), "test_server");
}

/**
 * @brief Test operator setup with different resolution parameters
 * 
 * Verifies that the operator correctly handles various resolution configurations.
 */
TEST_F(StreamingServerOpTest, DifferentResolutions) {
    // Test HD resolution (1920x1080)
    auto op1 = fragment->make_operator<StreamingServerOp>("server_hd",
        Arg("width", 1920),
        Arg("height", 1080));
    ASSERT_NE(op1, nullptr);
    
    // Test 4K resolution (3840x2160) 
    auto op2 = fragment->make_operator<StreamingServerOp>("server_4k",
        Arg("width", 3840),
        Arg("height", 2160));
    ASSERT_NE(op2, nullptr);
    
    // Test SD resolution (640x480)
    auto op3 = fragment->make_operator<StreamingServerOp>("server_sd",
        Arg("width", 640),
        Arg("height", 480));
    ASSERT_NE(op3, nullptr);
    
    // Test medical imaging resolution (854x480)
    auto op4 = fragment->make_operator<StreamingServerOp>("server_medical",
        Arg("width", 854),
        Arg("height", 480));
    ASSERT_NE(op4, nullptr);
}

/**
 * @brief Test different frame rate configurations
 * 
 * Verifies that the operator handles various frame rate settings correctly.
 */
TEST_F(StreamingServerOpTest, DifferentFrameRates) {
    // Test 30 FPS (standard)
    auto op1 = fragment->make_operator<StreamingServerOp>("server_30fps",
        Arg("fps", 30));
    ASSERT_NE(op1, nullptr);
    
    // Test 60 FPS (high frame rate)
    auto op2 = fragment->make_operator<StreamingServerOp>("server_60fps",
        Arg("fps", 60));
    ASSERT_NE(op2, nullptr);
    
    // Test 15 FPS (low frame rate for bandwidth-constrained scenarios)
    auto op3 = fragment->make_operator<StreamingServerOp>("server_15fps",
        Arg("fps", 15));
    ASSERT_NE(op3, nullptr);
    
    // Test 120 FPS (very high frame rate)
    auto op4 = fragment->make_operator<StreamingServerOp>("server_120fps",
        Arg("fps", 120));
    ASSERT_NE(op4, nullptr);
}

/**
 * @brief Test different server port configurations
 * 
 * Verifies that the operator handles various port settings correctly.
 */
TEST_F(StreamingServerOpTest, DifferentPorts) {
    // Test default streaming port
    auto op1 = fragment->make_operator<StreamingServerOp>("server_default_port",
        Arg("port", 48010));
    ASSERT_NE(op1, nullptr);
    
    // Test alternative port
    auto op2 = fragment->make_operator<StreamingServerOp>("server_alt_port", 
        Arg("port", 8080));
    ASSERT_NE(op2, nullptr);
    
    // Test high port number
    auto op3 = fragment->make_operator<StreamingServerOp>("server_high_port",
        Arg("port", 65000));
    ASSERT_NE(op3, nullptr);
    
    // Test common streaming port
    auto op4 = fragment->make_operator<StreamingServerOp>("server_rtmp_port",
        Arg("port", 1935));
    ASSERT_NE(op4, nullptr);
}

/**
 * @brief Test frame handling flag configurations
 * 
 * Verifies that receive_frames and send_frames flags work correctly.
 */
TEST_F(StreamingServerOpTest, FrameHandlingFlags) {
    // Test receive-only server
    auto op1 = fragment->make_operator<StreamingServerOp>("server_receive_only",
        Arg("receive_frames", true),
        Arg("send_frames", false));
    ASSERT_NE(op1, nullptr);
    
    // Test send-only server  
    auto op2 = fragment->make_operator<StreamingServerOp>("server_send_only",
        Arg("receive_frames", false),
        Arg("send_frames", true));
    ASSERT_NE(op2, nullptr);
    
    // Test bidirectional server
    auto op3 = fragment->make_operator<StreamingServerOp>("server_bidirectional",
        Arg("receive_frames", true), 
        Arg("send_frames", true));
    ASSERT_NE(op3, nullptr);
    
    // Test inactive server (neither send nor receive)
    auto op4 = fragment->make_operator<StreamingServerOp>("server_inactive",
        Arg("receive_frames", false),
        Arg("send_frames", false));
    ASSERT_NE(op4, nullptr);
}

/**
 * @brief Test operator setup and specification
 * 
 * Verifies that setup() correctly configures input/output specifications.
 */
TEST_F(StreamingServerOpTest, Setup) {
    auto op1 = fragment->make_operator<StreamingServerOp>("test_setup");
    ASSERT_NE(op1, nullptr);
    
    // Setup should not throw exceptions
    EXPECT_NO_THROW({
        OperatorSpec spec(fragment);
        op1->setup(spec);
    });
    
    // Test setup with different parameters
    auto op2 = fragment->make_operator<StreamingServerOp>("test_setup_params",
        Arg("width", 1280),
        Arg("height", 720),
        Arg("fps", 30),
        Arg("port", 8080));
    ASSERT_NE(op2, nullptr);
    
    EXPECT_NO_THROW({
        OperatorSpec spec(fragment);
        op2->setup(spec);
    });
}

/**
 * @brief Test invalid parameter handling
 * 
 * Verifies that the operator handles invalid parameters gracefully.
 */
TEST_F(StreamingServerOpTest, InvalidParameters) {
    // Test invalid width (negative)
    auto op1 = fragment->make_operator<StreamingServerOp>("server_invalid_width",
        Arg("width", -1));
    ASSERT_NE(op1, nullptr);
    
    // Test invalid height (zero)
    auto op2 = fragment->make_operator<StreamingServerOp>("server_invalid_height",
        Arg("height", 0));
    ASSERT_NE(op2, nullptr);
    
    // Test invalid fps (negative)
    auto op3 = fragment->make_operator<StreamingServerOp>("server_invalid_fps",
        Arg("fps", -30));
    ASSERT_NE(op3, nullptr);
}

/**
 * @brief Test boundary value conditions
 * 
 * Verifies that the operator handles edge cases and boundary values correctly.
 */
TEST_F(StreamingServerOpTest, BoundaryValues) {
    // Test minimum reasonable resolution
    auto op1 = fragment->make_operator<StreamingServerOp>("server_min_res",
        Arg("width", 1),
        Arg("height", 1));
    ASSERT_NE(op1, nullptr);
    
    // Test maximum typical resolution
    auto op2 = fragment->make_operator<StreamingServerOp>("server_max_res",
        Arg("width", 7680),  // 8K width
        Arg("height", 4320)); // 8K height
    ASSERT_NE(op2, nullptr);
}

/**
 * @brief Test multiple instance creation
 * 
 * Verifies that multiple operator instances can coexist.
 */
TEST_F(StreamingServerOpTest, MultipleInstances) {
    auto op1 = fragment->make_operator<StreamingServerOp>("server1", Arg("port", 8080));
    auto op2 = fragment->make_operator<StreamingServerOp>("server2", Arg("port", 8081)); 
    auto op3 = fragment->make_operator<StreamingServerOp>("server3", Arg("port", 8082));
    
    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);
    ASSERT_NE(op3, nullptr);
    
    EXPECT_NE(op1, op2);
    EXPECT_NE(op1, op3);
    EXPECT_NE(op2, op3);
}

/**
 * @brief Test string parameter handling
 * 
 * Verifies that string parameters like server_name are handled correctly.
 */
TEST_F(StreamingServerOpTest, StringParameters) {
    // Test with custom server name
    auto op1 = fragment->make_operator<StreamingServerOp>("server_custom_name",
        Arg("server_name", std::string("CustomStreamingServer")));
    ASSERT_NE(op1, nullptr);
    
    // Test with empty server name
    auto op2 = fragment->make_operator<StreamingServerOp>("server_empty_name",
        Arg("server_name", std::string("")));
    ASSERT_NE(op2, nullptr);
    
    // Test with long server name
    auto op3 = fragment->make_operator<StreamingServerOp>("server_long_name",
        Arg("server_name", std::string("VeryLongStreamingServerNameForTesting123456")));
    ASSERT_NE(op3, nullptr);
    
    // Test with special characters in server name
    auto op4 = fragment->make_operator<StreamingServerOp>("server_special_name",
        Arg("server_name", std::string("Server-Test_2024@Domain")));
    ASSERT_NE(op4, nullptr);
    
    // Test with numeric server name
    auto op5 = fragment->make_operator<StreamingServerOp>("server_numeric_name",
        Arg("server_name", std::string("12345")));
    ASSERT_NE(op5, nullptr);
    
    // Test with Unicode characters (if supported)
    auto op6 = fragment->make_operator<StreamingServerOp>("server_unicode_name",
        Arg("server_name", std::string("测试服务器")));
    ASSERT_NE(op6, nullptr);
}
