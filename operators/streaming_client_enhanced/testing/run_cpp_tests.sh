#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to manually build and run C++ unit tests for streaming_client_enhanced operator

set -e

echo "🏗️  Building streaming_client_enhanced operator with C++ tests..."

# Change to holohub root directory
cd /workspace/holohub

# Create a test-specific build directory
BUILD_DIR="build-streaming_client_enhanced-tests"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "📁 Build directory: $(pwd)"

# Configure with testing enabled
echo "⚙️  Configuring CMake with testing enabled..."
cmake \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D BUILD_TESTING=ON \
  -D BUILD_HOLOHUB_TESTING=ON \
  -D HOLOHUB_BUILD_OPERATORS=streaming_client_enhanced \
  -G Ninja \
  ..

echo "🔨 Building operator and tests..."
ninja streaming_client_enhanced_test

echo "🧪 Running C++ unit tests..."
if [ -f "./operators/streaming_client_enhanced/testing/streaming_client_enhanced_test" ]; then
    echo "✅ Found test executable: ./operators/streaming_client_enhanced/testing/streaming_client_enhanced_test"
    echo "🚀 Executing C++ tests..."
    ./operators/streaming_client_enhanced/testing/streaming_client_enhanced_test
    echo "✅ C++ tests completed successfully!"
else
    echo "❌ Test executable not found. Listing build output:"
    find . -name "*streaming*test*" -type f
    echo "📁 Current directory contents:"
    ls -la
    echo "📁 Operators directory:"
    find . -path "*/operators/streaming_client_enhanced*" -type d
fi

echo "🎉 C++ test execution completed!"
