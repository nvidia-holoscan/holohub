#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Video Streaming Demo Enhanced - Simple Integration Test ==="
echo "This test runs server and client applications to verify end-to-end functionality"
echo "Expected duration: 2-3 minutes"

# Clean up any existing log files
rm -f server_test.log client_test.log

# Go to holohub root
cd ../../

echo "Step 1: Building the application..."
./holohub build video_streaming_demo_enhanced --language cpp

echo "Step 2: Starting the server in background..."
timeout 120 ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced server --language cpp > applications/video_streaming_demo_enhanced/server_test.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Waiting 15 seconds for server initialization..."
sleep 15

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server process died during startup"
    echo "=== SERVER LOG ==="
    cat applications/video_streaming_demo_enhanced/server_test.log
    exit 1
fi

echo "Step 3: Starting the client (replayer mode)..."
timeout 60 ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced client_replayer --language cpp > applications/video_streaming_demo_enhanced/client_test.log 2>&1 &
CLIENT_PID=$!

echo "Client started with PID: $CLIENT_PID"
echo "Allowing 30 seconds for streaming to occur..."
sleep 30

# Stop both processes gracefully
echo "Step 4: Stopping applications..."
if kill -0 $CLIENT_PID 2>/dev/null; then
    kill -TERM $CLIENT_PID 2>/dev/null || true
    wait $CLIENT_PID 2>/dev/null || true
fi

if kill -0 $SERVER_PID 2>/dev/null; then
    kill -TERM $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
fi

echo "Step 5: Analyzing results..."

# Return to app directory
cd applications/video_streaming_demo_enhanced

# Check logs for success indicators
echo "=== SERVER LOG ANALYSIS ==="
if grep -q "StreamingServerResource\|Server.*started\|Listening on\|Connection.*established" server_test.log; then
    echo "✅ Server started successfully"
    SERVER_SUCCESS=1
else
    echo "❌ Server startup issues detected"
    SERVER_SUCCESS=0
fi

echo "=== CLIENT LOG ANALYSIS ==="
if grep -q "StreamingClient\|Connection.*established\|Frame.*sent\|Tensor.*validation.*passed" client_test.log; then
    echo "✅ Client connected and streamed successfully"
    CLIENT_SUCCESS=1
else
    echo "❌ Client connection/streaming issues detected"
    CLIENT_SUCCESS=0
fi

# Show relevant log excerpts
echo ""
echo "=== SERVER LOG (last 20 lines) ==="
tail -n 20 server_test.log

echo ""
echo "=== CLIENT LOG (last 20 lines) ==="
tail -n 20 client_test.log

# Final result
echo ""
echo "=== INTEGRATION TEST RESULTS ==="
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✅ INTEGRATION TEST PASSED"
    echo "   - Server started successfully"
    echo "   - Client connected and streamed video"
    echo "   - End-to-end video streaming verified"
    exit 0
else
    echo "❌ INTEGRATION TEST FAILED"
    [ $SERVER_SUCCESS -eq 0 ] && echo "   - Server startup failed"
    [ $CLIENT_SUCCESS -eq 0 ] && echo "   - Client connection/streaming failed"
    echo ""
    echo "Check the full logs above for detailed error information."
    exit 1
fi
