#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Integration Test ==="
echo "This test may take up to 10 minutes to complete..."

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Create a custom integration test that runs both server and client in the same container
echo "Creating custom integration test approach..."

# Go to holohub root if we're in the app directory
if [[ $(basename "$PWD") == "video_streaming_demo_enhanced" ]]; then
    cd ../../
fi

# Build the Docker image first
echo "Building Docker image for integration test..."
./holohub build video_streaming_demo_enhanced --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu --no-docker-build || {
    echo "Building Docker image..."
    ./holohub build video_streaming_demo_enhanced --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
}

# Create a custom integration test that runs both server and client in the same container
echo "Running custom integration test using holohub test..."

# Clean up build directory manually to avoid ctest_empty_binary_directory issues
echo "Cleaning up build directory manually..."
rm -rf build-video_streaming_demo_enhanced

# Use holohub test with the correct application name and test pattern
# Force Holoscan 3.5.0 to match our code requirements
./holohub test video_streaming_demo_enhanced --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming.*test" 2>&1 > applications/video_streaming_demo_enhanced/integration_test.log
INTEGRATION_EXIT_CODE=$?

# Check results
if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Integration test completed successfully"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
else
    echo "✗ Integration test failed with exit code $INTEGRATION_EXIT_CODE"
    SERVER_SUCCESS=0
    CLIENT_SUCCESS=0
fi

# Check the integration test log
echo "=== INTEGRATION TEST LOG ==="
cat applications/video_streaming_demo_enhanced/integration_test.log

# Verify success conditions
echo "=== VERIFICATION ==="

# Check integration test log for more detailed success indicators
if [ $INTEGRATION_EXIT_CODE -eq 0 ] && grep -q "Test.*Passed\|100%.*tests passed" applications/video_streaming_demo_enhanced/integration_test.log; then
    echo "✓ Integration test passed with detailed verification"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
elif [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Integration test completed successfully (basic verification)"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
else
    echo "✗ Integration test failed - checking for specific errors..."
    if grep -q "streaming.*server.*test" applications/video_streaming_demo_enhanced/integration_test.log; then
        echo "✗ Server test failed"
        SERVER_SUCCESS=0
    fi
    if grep -q "streaming.*client.*test" applications/video_streaming_demo_enhanced/integration_test.log; then
        echo "✗ Client test failed" 
        CLIENT_SUCCESS=0
    fi
fi

# Report individual component status
if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "✓ Server component verified"
else
    echo "✗ Server component failed"
fi

if [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Client component verified"
else
    echo "✗ Client component failed"
fi

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Integration test PASSED"
    exit 0
else
    echo "✗ Integration test FAILED"
    exit 1
fi
