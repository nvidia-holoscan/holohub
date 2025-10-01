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

# Create a custom test script that runs both server and client
cat > /tmp/integration_test_script.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Custom Integration Test ==="
echo "Starting server and client in same container..."

# Start server in background
echo "Starting streaming server..."
cd /workspace/holohub
./build-video_streaming_demo_enhanced/bin/streaming_server_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_server/cpp/streaming_server_demo.yaml &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 10

# Check if server is listening
echo "Checking if server is listening on port 48010..."
for i in {1..30}; do
    if netstat -tlnp 2>/dev/null | grep -q ":48010"; then
        echo "✓ Server is listening on port 48010"
        SERVER_SUCCESS=1
        break
    else
        echo "Waiting for server to start... (attempt $i/30)"
        sleep 2
    fi
done

if [ $SERVER_SUCCESS -eq 1 ]; then
    echo "✓ Server started successfully"
    
    # Run client test
    echo "Starting streaming client..."
    timeout 60s ./build-video_streaming_demo_enhanced/bin/streaming_client_demo_enhanced --config applications/video_streaming_demo_enhanced/video_streaming_demo_client/cpp/streaming_client_demo_replayer.yaml
    CLIENT_EXIT_CODE=$?
    
    if [ $CLIENT_EXIT_CODE -eq 0 ]; then
        echo "✓ Client test completed successfully"
        CLIENT_SUCCESS=1
    elif [ $CLIENT_EXIT_CODE -eq 124 ]; then
        echo "✗ Client test timed out"
        CLIENT_SUCCESS=0
    else
        echo "✗ Client test failed with exit code $CLIENT_EXIT_CODE"
        CLIENT_SUCCESS=0
    fi
else
    echo "✗ Server failed to start"
    CLIENT_SUCCESS=0
fi

# Clean up
echo "Cleaning up server process..."
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 2
kill -KILL $SERVER_PID 2>/dev/null || true

# Overall result
if [ $SERVER_SUCCESS -eq 1 ] && [ $CLIENT_SUCCESS -eq 1 ]; then
    echo "✓ Integration test PASSED"
    exit 0
else
    echo "✗ Integration test FAILED"
    exit 1
fi
EOF

chmod +x /tmp/integration_test_script.sh

# Run the custom integration test using holohub test with a custom script
echo "Running custom integration test..."
./holohub test video_streaming_demo_enhanced --cmake-options="-DBUILD_TESTING=ON" --ctest-options="-S /tmp/integration_test_script.sh" 2>&1 > integration_test.log
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

# Clean up temporary script
rm -f /tmp/integration_test_script.sh

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
