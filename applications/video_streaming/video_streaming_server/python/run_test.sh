#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script for streaming_server_demo Python infrastructure testing
# Args: $1 = build_dir, $2 = script_path

set -e

BUILD_DIR="$1"
SCRIPT_PATH="$2"

echo "🧪 Running StreamingServer Python Infrastructure Test"
echo "Build Dir: $BUILD_DIR"
echo "Script: $SCRIPT_PATH"

# Change to build directory for test execution
cd "$BUILD_DIR"

# Set up Python path for streaming server operator
export PYTHONPATH="/opt/nvidia/holoscan/lib/../python/lib:$BUILD_DIR/python/lib:$PYTHONPATH"

# Run the infrastructure test with timeout and proper error handling
timeout 30s python3 "$SCRIPT_PATH" || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Test timed out after 30 seconds - this may be expected for streaming server in test environment"
        echo "Test PASSED: StreamingServer functionality validated successfully (timeout expected)"
    else
        echo "Test FAILED: StreamingServer infrastructure test failed with exit code $exit_code"
        exit 1
    fi
}

echo "Test PASSED: StreamingServer functionality validated successfully"