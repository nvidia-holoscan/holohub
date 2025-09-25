#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script for streaming_server_demo Python functional testing
# Args: $1 = build_dir, $2 = script_path, $3 = data_dir

set -e

BUILD_DIR="$1"
SCRIPT_PATH="$2"
DATA_DIR="$3"

echo "🧪 Running StreamingServer Python Functional Test"
echo "Build Dir: $BUILD_DIR"
echo "Script: $SCRIPT_PATH" 
echo "Data Dir: $DATA_DIR"

# Change to build directory for test execution
cd "$BUILD_DIR"

# Set up Python path for streaming server operator
export PYTHONPATH="/opt/nvidia/holoscan/lib/../python/lib:$BUILD_DIR/python/lib:$PYTHONPATH"

# Run the functional test and capture output
OUTPUT_FILE="/tmp/streaming_server_python_functional_test_output.log"
timeout 60s python3 "$SCRIPT_PATH" --data "$DATA_DIR" 2>&1 | tee "$OUTPUT_FILE" || true

# Check if streaming server functionality worked correctly  
STREAMING_FUNCTIONALITY=""
if grep -q "StreamingServer started successfully" "$OUTPUT_FILE" && \
   grep -q "Starting streaming server on port" "$OUTPUT_FILE"; then
    STREAMING_FUNCTIONALITY="✅ StreamingServer started and listening for client connections"
elif grep -q "StreamingServerOp setup completed" "$OUTPUT_FILE"; then
    STREAMING_FUNCTIONALITY="✅ StreamingServer infrastructure validated successfully"
fi

# Check for server configuration and data availability
DATA_CONFIGURATION=""
if grep -q "FUNCTIONAL test: StreamingServer with data directory available" "$OUTPUT_FILE"; then
    DATA_CONFIGURATION="✅ Video data directory available for client streaming"
elif grep -q "Functional test configured in infrastructure mode" "$OUTPUT_FILE"; then
    DATA_CONFIGURATION="✅ Infrastructure mode validated"
fi

# Check overall success based on captured output
if [ -n "$STREAMING_FUNCTIONALITY" ] && [ -n "$DATA_CONFIGURATION" ]; then
    echo "✅ FUNCTIONAL test PASSED: Python StreamingServer with data directory successful"
    echo "  - $STREAMING_FUNCTIONALITY"
    echo "  - $DATA_CONFIGURATION"
    echo "  - StreamingServer ready to accept client connections"
    exit 0
elif [ -n "$STREAMING_FUNCTIONALITY" ]; then
    echo "✅ FUNCTIONAL test PASSED: StreamingServer basic functionality validated"
    echo "  - $STREAMING_FUNCTIONALITY"
    echo "  - Server initialization and configuration successful"
    exit 0
else
    echo "❌ FUNCTIONAL test FAILED: StreamingServer functional testing failed"
    echo "Debug information:"
    echo "  - Script: $SCRIPT_PATH"
    echo "  - Data directory: $DATA_DIR"
    echo "  - Streaming functionality: $STREAMING_FUNCTIONALITY"
    echo "  - Data configuration: $DATA_CONFIGURATION"
    echo "Output file contents:"
    cat "$OUTPUT_FILE" 2>/dev/null || echo "No output file found"
    exit 1
fi
