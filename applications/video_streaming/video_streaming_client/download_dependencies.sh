#!/bin/bash
# Setup script for Video Streaming Client dependencies
# This script downloads the required client streaming binaries from NGC

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Detect holohub root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOHUB_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
print_info "Detected HoloHub root: $HOLOHUB_ROOT"

# Check if ngc CLI is installed
if ! command -v ngc &> /dev/null; then
    print_error "NGC CLI is not installed or not in PATH"
    print_info "Please install NGC CLI from: https://ngc.nvidia.com/setup/installers/cli"
    exit 1
fi

print_info "NGC CLI found: $(which ngc)"

# Client dependencies
print_info "=========================================="
print_info "Downloading Client Dependencies..."
print_info "=========================================="

CLIENT_DIR="$HOLOHUB_ROOT/operators/video_streaming/video_streaming_client"
if [ ! -d "$CLIENT_DIR" ]; then
    print_error "Client directory not found: $CLIENT_DIR"
    exit 1
fi

cd "$CLIENT_DIR"
print_info "Working directory: $(pwd)"

print_info "Downloading holoscan_client_cloud_streaming:0.2 from NGC..."
if ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"; then
    print_info "Download successful"
else
    print_error "Failed to download client dependencies"
    exit 1
fi

print_info "Extracting client binaries..."
if unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming; then
    print_info "Extraction successful"
else
    print_error "Failed to extract client binaries"
    exit 1
fi

print_info "Cleaning up temporary files..."
rm -rf holoscan_client_cloud_streaming_v0.2

# Summary
print_info ""
print_info "=========================================="
print_info "Setup Complete!"
print_info "=========================================="
print_info "Client dependencies have been downloaded and installed."
print_info "Location: $CLIENT_DIR/holoscan_client_cloud_streaming"
print_info ""
print_info "You can now run the video streaming client:"
print_info "  - V4L2 mode:     ./holohub run video_streaming_client v4l2"
print_info "  - Replayer mode: ./holohub run video_streaming_client replayer"

