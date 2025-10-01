#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Sequential Integration Test ==="

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Test 1: Run server test
echo "=== Test 1: Server Test ==="
echo "Starting streaming server test..."
timeout 60s ../../../holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_server_demo_enhanced_cpp_test" 2>&1 > streamingserver.log &
SERVER_PID=$!

# Wait for server test to complete
wait $SERVER_PID
SERVER_EXIT_CODE=$?

echo "Server test completed with exit code: $SERVER_EXIT_CODE"

# Check server test results
echo "=== SERVER LOG ==="
cat streamingserver.log

# Test 2: Run client test
echo "=== Test 2: Client Test ==="
echo "Starting streaming client test..."
timeout 60s ../../../holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_client_demo_enhanced_cpp_test" 2>&1 > streamingclient.log &
CLIENT_PID=$!

# Wait for client test to complete
wait $CLIENT_PID
CLIENT_EXIT_CODE=$?

echo "Client test completed with exit code: $CLIENT_EXIT_CODE"

# Check client test results
echo "=== CLIENT LOG ==="
cat streamingclient.log

# Verify success conditions
echo "=== VERIFICATION ==="

# Check server test success
if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo "✓ Server test PASSED"
    SERVER_SUCCESS=1
else
    echo "✗ Server test FAILED (exit code: $SERVER_EXIT_CODE)"
    SERVER_SUCCESS=0
fi

# Check client test success
if [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "✓ Client test PASSED"
    CLIENT_SUCCESS=1
else
    echo "✗ Client test FAILED (exit code: $CLIENT_EXIT_CODE)"
    CLIENT_SUCCESS=0
fi

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Sequential integration test PASSED"
    exit 0
else
    echo "✗ Sequential integration test FAILED"
    exit 1
fi
