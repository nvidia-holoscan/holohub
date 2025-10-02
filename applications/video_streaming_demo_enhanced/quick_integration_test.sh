#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=== Quick Integration Test (Using Existing Build) ==="
echo "This test uses existing builds to avoid compilation issues"
echo "Expected duration: 1-2 minutes"

# Clean up any existing log files
rm -f server_quick_test.log client_quick_test.log

# Go to holohub root
cd ../../

echo "Step 1: Starting the server in background..."
timeout 120 ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced server --language cpp > applications/video_streaming_demo_enhanced/server_quick_test.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Waiting 20 seconds for server initialization..."
sleep 20

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server process died during startup"
    echo "=== SERVER LOG ==="
    cat applications/video_streaming_demo_enhanced/server_quick_test.log
    exit 1
fi

echo "Step 2: Starting the client (replayer mode)..."
timeout 60 ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced client_replayer --language cpp > applications/video_streaming_demo_enhanced/client_quick_test.log 2>&1 &
CLIENT_PID=$!

echo "Client started with PID: $CLIENT_PID"
echo "Allowing 20 seconds for streaming to occur..."
sleep 20

# Stop both processes gracefully
echo "Step 3: Stopping applications..."
if kill -0 $CLIENT_PID 2>/dev/null; then
    kill -TERM $CLIENT_PID 2>/dev/null || true
    wait $CLIENT_PID 2>/dev/null || true
fi

if kill -0 $SERVER_PID 2>/dev/null; then
    kill -TERM $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
fi

echo "Step 4: Analyzing results..."

# Return to app directory
cd applications/video_streaming_demo_enhanced

# Check logs for success indicators
echo "=== SERVER LOG ANALYSIS ==="
if grep -q "StreamingServerResource.*started\|Server.*listening\|started successfully" server_quick_test.log; then
    echo "✅ Server started successfully"
    SERVER_SUCCESS=1
else
    echo "❌ Server startup issues detected"
    SERVER_SUCCESS=0
fi

echo "=== CLIENT LOG ANALYSIS ==="
if grep -q "StreamingClient.*created\|Connection.*established\|started successfully" client_quick_test.log; then
    echo "✅ Client connected successfully"
    CLIENT_SUCCESS=1
else
    echo "❌ Client connection issues detected"
    CLIENT_SUCCESS=0
fi

# Show relevant log excerpts
echo ""
echo "=== SERVER LOG (last 15 lines) ==="
tail -n 15 server_quick_test.log

echo ""
echo "=== CLIENT LOG (last 15 lines) ==="
tail -n 15 client_quick_test.log

# Final result
echo ""
echo "=== QUICK INTEGRATION TEST RESULTS ==="
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✅ INTEGRATION TEST PASSED"
    echo "   - Server started successfully"
    echo "   - Client connected successfully"
    echo "   - Basic functionality verified"
    exit 0
else
    echo "❌ INTEGRATION TEST FAILED"
    [ $SERVER_SUCCESS -eq 0 ] && echo "   - Server startup failed"
    [ $CLIENT_SUCCESS -eq 0 ] && echo "   - Client connection failed"
    echo ""
    echo "Check the full logs above for detailed error information."
    exit 1
fi
