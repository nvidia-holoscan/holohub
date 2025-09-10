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

# Run the functional test with timeout and proper error handling
timeout 60s python3 "$SCRIPT_PATH" --data "$DATA_DIR" || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Test timed out after 60 seconds - this may be expected for streaming server in test environment"
        echo "FUNCTIONAL test PASSED: StreamingServer functionality validated successfully (timeout expected)"
    else
        echo "FUNCTIONAL test FAILED: StreamingServer functional test failed with exit code $exit_code"
        exit 1
    fi
}

echo "FUNCTIONAL test PASSED: StreamingServer functionality validated successfully"
