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

# Build the Docker image first
echo "Building Docker image for integration test..."
./holohub build video_streaming_demo_enhanced --no-docker-build || {
    echo "Building Docker image..."
    ./holohub build video_streaming_demo_enhanced
}

# Create a custom integration test that runs both server and client in the same container
echo "Running custom integration test using holohub test..."

# Use holohub test with a custom command that runs both server and client
./holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R video_streaming_integration_test" 2>&1 > integration_test.log
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

# Check the log files
echo "=== SERVER LOG ==="
cat streamingserver.log

echo "=== CLIENT LOG ==="
cat streamingclient.log

# Verify success conditions
echo "=== VERIFICATION ==="

# Server success is already determined by exit code
if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "✓ Server test passed"
else
    echo "✗ Server test failed"
fi

# Client success is already determined by exit code
if [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Client test passed"
else
    echo "✗ Client test failed"
fi

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Integration test PASSED"
    exit 0
else
    echo "✗ Integration test FAILED"
    exit 1
fi
