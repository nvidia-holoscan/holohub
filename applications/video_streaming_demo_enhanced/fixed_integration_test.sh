#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Fixed Integration Test ==="

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Test 1: Run client test (since server test doesn't exist in main CMakeLists)
echo "=== Test 1: Client Test ==="
echo "Starting streaming client test..."
timeout 60s /workspace/holohub/holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_client_demo_enhanced_cpp_test" 2>&1 > streamingclient.log &
CLIENT_PID=$!

# Wait for client test to complete
wait $CLIENT_PID
CLIENT_EXIT_CODE=$?

echo "Client test completed with exit code: $CLIENT_EXIT_CODE"

# Check client test results
echo "=== CLIENT LOG ==="
cat streamingclient.log

# Test 2: Run integration test (the actual integration test script)
echo "=== Test 2: Integration Test ==="
echo "Starting integration test..."
timeout 60s /workspace/holohub/holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R video_streaming_integration_test" 2>&1 > streamingserver.log &
INTEGRATION_PID=$!

# Wait for integration test to complete
wait $INTEGRATION_PID
INTEGRATION_EXIT_CODE=$?

echo "Integration test completed with exit code: $INTEGRATION_EXIT_CODE"

# Check integration test results
echo "=== INTEGRATION LOG ==="
cat streamingserver.log

# Verify success conditions
echo "=== VERIFICATION ==="

# Check client test success
if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "✓ Client test PASSED"
    CLIENT_SUCCESS=1
else
    echo "✗ Client test FAILED (exit code: $CLIENT_EXIT_CODE)"
    CLIENT_SUCCESS=0
fi

# Check integration test success
if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo "✓ Integration test PASSED"
    INTEGRATION_SUCCESS=1
else
    echo "✗ Integration test FAILED (exit code: $INTEGRATION_EXIT_CODE)"
    INTEGRATION_SUCCESS=0
fi

# Overall result
if [ $CLIENT_SUCCESS -eq 1 ] && [ $INTEGRATION_SUCCESS -eq 1 ]; then
    echo "✓ Fixed integration test PASSED"
    exit 0
else
    echo "✗ Fixed integration test FAILED"
    exit 1
fi
