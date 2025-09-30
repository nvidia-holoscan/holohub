#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Integration Test ==="

# Clean up any existing log files
rm -f streamingserver.log streamingclient.log

# Launch server
echo "Starting streaming server..."
./holohub test --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced/video_streaming_demo_server --language cpp 2>&1 > streamingserver.log &
SERVER_PID=$!

sleep 10

# Launch client with replayer config
echo "Starting streaming client..."
./holohub test --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced/video_streaming_demo_client --language cpp --run-args='-c streaming_client_demo_replayer.yaml' 2>&1 > streamingclient.log &
CLIENT_PID=$!

sleep 30

# Wait for both server and client to terminate (timeout = 10 secs)
echo "Waiting for processes to complete..."
timeout 10s bash -c "wait $SERVER_PID $CLIENT_PID" || echo "Timeout reached, killing processes..."

# Kill processes if still running
kill -9 $SERVER_PID $CLIENT_PID 2>/dev/null || true

# Check the log files
echo "=== SERVER LOG ==="
cat streamingserver.log

echo "=== CLIENT LOG ==="
cat streamingclient.log

# Verify success conditions
echo "=== VERIFICATION ==="

# Check server started successfully
if grep -q "StreamingServerResource started successfully\|Server started\|Listening on" streamingserver.log; then
    echo "✓ Server started successfully"
    SERVER_SUCCESS=1
else
    echo "✗ Server failed to start"
    SERVER_SUCCESS=0
fi

# Check client connected/created successfully
if grep -q "StreamingClient created successfully\|Client connected\|Connection established" streamingclient.log; then
    echo "✓ Client created/connected successfully"
    CLIENT_SUCCESS=1
else
    echo "✗ Client failed to create/connect"
    CLIENT_SUCCESS=0
fi

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Integration test PASSED"
    exit 0
else
    echo "✗ Integration test FAILED"
    exit 1
fi
