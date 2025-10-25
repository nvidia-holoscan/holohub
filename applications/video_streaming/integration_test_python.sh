#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python Integration Test for Video Streaming Demo
# 
# IMPORTANT: This test runs in Docker and builds from committed source code.
# If you have local Python/C++ changes, make sure they are committed before running this test.

set -e

echo "=== Python Video Streaming Demo Integration Test ==="
echo "This test may take up to 10 minutes to complete..."
echo "NOTE: Test runs in Docker and uses committed source code (not local build)"

# Clean up any existing log files
rm -f applications/video_streaming/integration_test_python.log

# Go to holohub root if we're in the app directory
if [[ $(basename "$PWD") == "video_streaming" ]]; then
    cd ../../
fi

# Ensure we're using the latest committed changes
echo "Forcing Docker to use latest committed changes..."
echo "Current commit: $(git log --oneline -1)"

# Clean up any cached Docker builds to ensure fresh build with latest commits
echo "Cleaning Docker build cache..."
docker system prune -f --filter "label=holohub" 2>/dev/null || true

# Build and test using Docker with Python support (this will use your committed fixes)
echo "Running Python integration test with Docker (using committed fixes)..."
# Set SDK version via environment variable to match base image version
export HOLOHUB_BASE_SDK_VERSION=3.6.0
./holohub test video_streaming \
    --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.0-dgpu \
    --cmake-options="-DBUILD_TESTING=ON -DHOLOHUB_BUILD_PYTHON=ON" \
    --ctest-options="-R video_streaming_integration_test_python -V" \
    --verbose 2>&1 | tee applications/video_streaming/integration_test_python.log
INTEGRATION_EXIT_CODE=$?

# Check results
if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Python Integration test completed successfully"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
else
    echo "✗ Python Integration test failed with exit code $INTEGRATION_EXIT_CODE"
    SERVER_SUCCESS=0
    CLIENT_SUCCESS=0
fi

# Check the integration test log
echo "=== PYTHON INTEGRATION TEST LOG ==="
if [ -f applications/video_streaming/integration_test_python.log ]; then
    cat applications/video_streaming/integration_test_python.log
else
    echo "Warning: Log file not found (test may have failed before logging)"
fi

# Verify success conditions
echo "=== VERIFICATION ==="

# Check integration test log for more detailed success indicators
if [ $INTEGRATION_EXIT_CODE -eq 0 ] && [ -f applications/video_streaming/integration_test_python.log ] && grep -qE "Python Integration test PASSED|100% tests passed, 0 tests failed" applications/video_streaming/integration_test_python.log; then
    echo "✓ Python integration test passed with detailed verification"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
elif [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Python integration test completed successfully (basic verification)"
    SERVER_SUCCESS=1
    CLIENT_SUCCESS=1
else
    echo "✗ Python integration test failed - checking for specific errors..."
    if [ -f applications/video_streaming/integration_test_python.log ]; then
        if grep -qE "Python server.*failed|server_python.log" applications/video_streaming/integration_test_python.log; then
            echo "✗ Python server test failed"
            SERVER_SUCCESS=0
        fi
        if grep -qE "Python client.*failed|client_python.log" applications/video_streaming/integration_test_python.log; then
            echo "✗ Python client test failed" 
            CLIENT_SUCCESS=0
        fi
    else
        echo "✗ Cannot check specific errors - log file not found"
    fi
fi

# Report individual component status
if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "✓ Python server component verified"
else
    echo "✗ Python server component failed"
fi

if [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Python client component verified"
else
    echo "✗ Python client component failed"
fi

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Python Integration test PASSED"
    exit 0
else
    echo "✗ Python Integration test FAILED"
    exit 1
fi

