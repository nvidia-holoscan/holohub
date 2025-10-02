#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Simple Local Integration Test ==="
echo "This test uses your local build with all fixes applied"

# Clean up any existing log files
rm -f server_local_test.log client_local_test.log

# Go to holohub root if we're in the app directory
if [[ $(basename "$PWD") == "video_streaming_demo_enhanced" ]]; then
    cd ../../
fi

echo "Testing server startup..."
timeout 30s ./holohub run video_streaming_demo_enhanced server --language cpp --local > applications/video_streaming_demo_enhanced/server_local_test.log 2>&1 &
SERVER_PID=$!

# Wait a bit for server to start
sleep 5

# Check if server is still running (not crashed)
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✓ Server started successfully"
    SERVER_SUCCESS=1
else
    echo "✗ Server failed to start or crashed"
    SERVER_SUCCESS=0
fi

# Stop the server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Testing client startup..."
timeout 30s ./holohub run video_streaming_demo_enhanced client_replayer --language cpp --local > applications/video_streaming_demo_enhanced/client_local_test.log 2>&1 &
CLIENT_PID=$!

# Wait a bit for client to start
sleep 5

# Check if client is still running (not crashed)
if kill -0 $CLIENT_PID 2>/dev/null; then
    echo "✓ Client started successfully"
    CLIENT_SUCCESS=1
else
    echo "✗ Client failed to start or crashed"
    CLIENT_SUCCESS=0
fi

# Stop the client
kill $CLIENT_PID 2>/dev/null || true
wait $CLIENT_PID 2>/dev/null || true

# Analyze logs for success indicators
echo "=== LOG ANALYSIS ==="

echo "Server log analysis:"
if grep -q "StreamingServer.*started\|Server.*running\|Listening.*port" applications/video_streaming_demo_enhanced/server_local_test.log; then
    echo "✓ Server log shows successful startup"
    SERVER_LOG_SUCCESS=1
else
    echo "✗ Server log does not show successful startup"
    SERVER_LOG_SUCCESS=0
fi

echo "Client log analysis:"
if grep -q "StreamingClient.*created\|Client.*connected\|Successfully.*initialized" applications/video_streaming_demo_enhanced/client_local_test.log; then
    echo "✓ Client log shows successful startup"
    CLIENT_LOG_SUCCESS=1
else
    echo "✗ Client log does not show successful startup"
    CLIENT_LOG_SUCCESS=0
fi

# Show relevant log excerpts
echo "=== SERVER LOG EXCERPT ==="
head -20 applications/video_streaming_demo_enhanced/server_local_test.log

echo "=== CLIENT LOG EXCERPT ==="
head -20 applications/video_streaming_demo_enhanced/client_local_test.log

# Final assessment
OVERALL_SUCCESS=1

if [ $SERVER_SUCCESS -eq 1 ] && [ $SERVER_LOG_SUCCESS -eq 1 ]; then
    echo "✓ Server component PASSED"
else
    echo "✗ Server component FAILED"
    OVERALL_SUCCESS=0
fi

if [ $CLIENT_SUCCESS -eq 1 ] && [ $CLIENT_LOG_SUCCESS -eq 1 ]; then
    echo "✓ Client component PASSED"
else
    echo "✗ Client component FAILED"
    OVERALL_SUCCESS=0
fi

if [ $OVERALL_SUCCESS -eq 1 ]; then
    echo "✓ Simple Local Integration Test PASSED"
    exit 0
else
    echo "✗ Simple Local Integration Test FAILED"
    exit 1
fi
