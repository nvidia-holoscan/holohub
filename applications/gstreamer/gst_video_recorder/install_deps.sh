# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# install_deps.sh - Install dependencies for gst_video_recorder local development
# 
# This script installs all required dependencies for building gst_video_recorder
# locally (with --local flag) instead of using the containerized build.
#
# Usage: ./install_deps.sh

set -e  # Exit on any error

echo "🔧 Installing GStreamer Video Recorder dependencies..."

# Update package lists
echo "📦 Updating package lists..."
sudo apt-get update

# Install pkg-config (required for CMake to find GStreamer)
echo "📦 Installing pkg-config..."
sudo apt-get install -y pkg-config

# Install GStreamer development libraries
echo "📦 Installing GStreamer development packages..."
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev

# Install GStreamer plugins for comprehensive codec support
echo "📦 Installing GStreamer plugins..."
sudo apt-get install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# Try to install optional CUDA plugin (requires GStreamer 1.24+)
echo "🚀 Installing optional CUDA support (if available)..."
if sudo apt-get install -y gstreamer1.0-cuda 2>/dev/null; then
    echo "✅ CUDA support installed successfully"
else
    echo "⚠️  CUDA support not available (requires GStreamer 1.24+)"
    echo "    This is normal for older GStreamer versions - CUDA features will be disabled"
fi

# Clean up
echo "🧹 Cleaning up..."
sudo apt-get autoremove -y
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "✅ All dependencies installed successfully!"
echo ""
echo "You can now build gst_video_recorder locally with:"
echo "  ./holohub build --local gst_video_recorder"
echo ""
echo "Or use the containerized build (no setup required):"
echo "  ./holohub build gst_video_recorder"
