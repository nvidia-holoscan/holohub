#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Integration Test ==="

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Launch server first and wait for it to complete
echo "Starting streaming server..."
./holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_server_demo_enhanced_cpp_test" 2>&1 > streamingserver.log
SERVER_EXIT_CODE=$?

# Check if server test passed
if [ $SERVER_EXIT_CODE -eq 0 ]; then
    echo "✓ Server test completed successfully"
    SERVER_SUCCESS=1
else
    echo "✗ Server test failed with exit code $SERVER_EXIT_CODE"
    SERVER_SUCCESS=0
fi

# Wait a bit for server to fully initialize
echo "Waiting for server to initialize..."
sleep 5

# Launch client only if server was successful
if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "Starting streaming client..."
    ./holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_client_demo_enhanced_cpp_test" 2>&1 > streamingclient.log
    CLIENT_EXIT_CODE=$?
    
    # Check if client test passed
    if [ $CLIENT_EXIT_CODE -eq 0 ]; then
        echo "✓ Client test completed successfully"
        CLIENT_SUCCESS=1
    else
        echo "✗ Client test failed with exit code $CLIENT_EXIT_CODE"
        CLIENT_SUCCESS=0
    fi
else
    echo "Skipping client test due to server failure"
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
