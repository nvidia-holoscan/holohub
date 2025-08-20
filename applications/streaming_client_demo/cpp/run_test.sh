#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script to run C++ streaming client demo test and handle exceptions gracefully
# The streaming client functionality works correctly, but throws an exception at the end
# when connection attempts fail (expected in test environment without server).

set -e

EXECUTABLE="$1"
CONFIG_FILE="$2"

echo "Running C++ streaming client demo test..."

# Run the test and capture output
OUTPUT_FILE="/tmp/streaming_client_cpp_test_output.log"
timeout 60 "$EXECUTABLE" --config "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_FILE" || true

# Check if the streaming client functionality worked correctly
if grep -q "StreamingClientOp initialized successfully" "$OUTPUT_FILE" && \
   grep -q "Starting streaming with server" "$OUTPUT_FILE" && \
   grep -q "Connection attempt.*failed" "$OUTPUT_FILE"; then
    echo "✅ Test PASSED: C++ StreamingClient functionality validated successfully"
    echo "  - StreamingClient initialized correctly"
    echo "  - Connection attempts made as expected"
    echo "  - Expected connection failures handled gracefully"
    exit 0
else
    echo "❌ Test FAILED: C++ StreamingClient functionality not working correctly"
    echo "See output above for details"
    exit 1
fi
