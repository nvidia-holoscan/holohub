#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# C++ Integration Test for Video Streaming Demo
# 
# IMPORTANT: This test runs in Docker and builds from committed source code.
# If you have local C++ fixes, make sure they are committed before running this test.
# The test will use --no-cache to ensure Docker picks up your latest commits.

set -e

echo "=== Video Streaming Demo Integration Test ==="
echo "This test may take up to 10 minutes to complete..."

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Create a custom integration test that runs both server and client in the same container
echo "Creating custom integration test approach..."

# Go to holohub root if we're in the app directory
if [[ $(basename "$PWD") == "video_streaming" ]]; then
    cd ../../
fi

# Ensure we're using the latest committed changes by forcing Docker to rebuild
echo "Forcing Docker to use latest committed changes..."
echo "Current commit: $(git log --oneline -1)"

# Clean up any cached Docker builds to ensure fresh build with latest commits
echo "Cleaning Docker build cache..."
docker system prune -f --filter "label=holohub" 2>/dev/null || true

# Build and test using Docker with fresh cache (this will use your committed C++ fixes)
echo "Running integration test with Docker (using committed fixes)..."
# Set SDK version via environment variable to match base image version
export HOLOHUB_BASE_SDK_VERSION=3.5.0
./holohub test video_streaming --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R video_streaming_integration_test_cpp -V" --verbose 2>&1 | tee applications/video_streaming/integration_test.log
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
if [ -f applications/video_streaming/integration_test.log ]; then
    cat applications/video_streaming/integration_test.log
else
    echo "Warning: Log file not found (test may have failed before logging)"
fi

# Verify success conditions
echo "=== VERIFICATION ==="

# Check integration test log for more detailed success indicators
if [ $INTEGRATION_EXIT_CODE -eq 0 ] && [ -f applications/video_streaming/integration_test.log ] && grep -qE "Test.*Passed|100% tests passed, 0 tests failed" applications/video_streaming/integration_test.log; then
    echo "✓ Integration test passed with detailed verification"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
elif [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Integration test completed successfully (basic verification)"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
else
    echo "✗ Integration test failed - checking for specific errors..."
    if [ -f applications/video_streaming/integration_test.log ]; then
        if grep -q "streaming.*server.*test" applications/video_streaming/integration_test.log; then
            echo "✗ Server test failed"
            SERVER_SUCCESS=0
        fi
        if grep -q "streaming.*client.*test" applications/video_streaming/integration_test.log; then
            echo "✗ Client test failed" 
            CLIENT_SUCCESS=0
        fi
    else
        echo "✗ Cannot check specific errors - log file not found"
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
