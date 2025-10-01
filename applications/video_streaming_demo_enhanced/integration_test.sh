#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Integration Test ==="

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Launch server in background and let it run continuously
echo "Starting streaming server in background..."
./holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-R streaming_server_demo_enhanced_cpp_test" 2>&1 > streamingserver.log &
SERVER_PID=$!

# Wait for server to initialize and start listening
echo "Waiting for server to initialize..."
sleep 10

# Check if server is listening on port 48010
echo "Checking if server is listening on port 48010..."
for i in {1..30}; do
    if netstat -tlnp 2>/dev/null | grep -q ":48010"; then
        echo "✓ Server is listening on port 48010"
        SERVER_SUCCESS=1
        break
    elif lsof -i :48010 2>/dev/null | grep -q LISTEN; then
        echo "✓ Server is listening on port 48010 (lsof)"
        SERVER_SUCCESS=1
        break
    else
        echo "Waiting for server to start... (attempt $i/30)"
        sleep 2
    fi
done

# Check if server started successfully
if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "✓ Server started successfully and is listening"
else
    echo "✗ Server failed to start or is not listening on port 48010"
    SERVER_SUCCESS=0
fi

# Launch client only if server is running and listening
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

# Clean up: kill server process
echo "Cleaning up server process..."
if [ ! -z "$SERVER_PID" ]; then
    kill -TERM $SERVER_PID 2>/dev/null || true
    sleep 2
    kill -KILL $SERVER_PID 2>/dev/null || true
    echo "✓ Server process cleaned up"
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
