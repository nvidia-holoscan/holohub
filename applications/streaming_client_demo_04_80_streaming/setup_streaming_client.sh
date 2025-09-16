#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPERATOR_DIR="${SCRIPT_DIR}/../../operators/streaming_client_04_80"
NGC_RESOURCE="nvstaging/holoscan/holoscan_client_cloud_streaming:1.0"
DOWNLOAD_DIR="holoscan_client_cloud_streaming_v1.0"
EXTRACT_DIR="holoscan_client_cloud_streaming"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🚀 Holoscan Streaming Client Setup Script"
    echo "=================================================="
    echo -e "${NC}"
}

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if NGC CLI is installed
    if ! command -v ngc &> /dev/null; then
        print_error "NGC CLI not found. Please install NGC CLI first:"
        echo "  wget --content-disposition https://ngc.nvidia.com/cli/ngccli_linux.zip"
        echo "  unzip ngccli_linux.zip"
        echo "  chmod u+x ngc-cli/ngc"
        echo "  sudo mv ngc-cli/ngc /usr/local/bin/"
        echo ""
        echo "Then configure NGC with your API key:"
        echo "  ngc config set"
        exit 1
    fi
    
    # Check if NGC is configured
    if ! ngc config current &> /dev/null; then
        print_error "NGC CLI not configured. Please run: ngc config set"
        exit 1
    fi
    
    # Check if we're in the right directory structure
    if [[ ! -d "$OPERATOR_DIR" ]]; then
        print_error "Operator directory not found: $OPERATOR_DIR"
        print_info "Please run this script from the holohub-internal root or streaming_client_demo application directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

detect_architecture() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            ARCH_DIR="x86_64"
            ;;
        aarch64)
            ARCH_DIR="aarch64"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            print_info "Supported architectures: x86_64, aarch64"
            exit 1
            ;;
    esac
    print_info "Detected architecture: $ARCH_DIR"
}

download_client_binaries() {
    print_info "Downloading streaming client binaries from NGC..."
    
    cd "$OPERATOR_DIR"
    
    # Clean up any existing downloads
    if [[ -d "$DOWNLOAD_DIR" ]]; then
        print_warning "Removing existing download directory: $DOWNLOAD_DIR"
        rm -rf "$DOWNLOAD_DIR"
    fi
    
    if [[ -f "holoscan_client_cloud_streaming.zip" ]]; then
        print_warning "Removing existing zip file"
        rm -f "holoscan_client_cloud_streaming.zip"
    fi
    
    # Download from NGC
    print_info "Running: ngc registry resource download-version \"$NGC_RESOURCE\""
    if ngc registry resource download-version "$NGC_RESOURCE"; then
        print_success "Download completed successfully"
    else
        print_error "Failed to download from NGC. Please check:"
        echo "  - Your NGC CLI configuration (ngc config current)"
        echo "  - Your access permissions to the resource"
        echo "  - Network connectivity"
        exit 1
    fi
}

extract_and_setup() {
    print_info "Extracting and setting up client binaries..."
    
    cd "$OPERATOR_DIR"
    
    # Extract the zip file
    if [[ -f "$DOWNLOAD_DIR/holoscan_client_cloud_streaming.zip" ]]; then
        print_info "Extracting holoscan_client_cloud_streaming.zip..."
        unzip -o "$DOWNLOAD_DIR/holoscan_client_cloud_streaming.zip"
        print_success "Extraction completed"
    else
        print_error "Zip file not found: $DOWNLOAD_DIR/holoscan_client_cloud_streaming.zip"
        exit 1
    fi
    
    # Check if extraction created the expected directory structure
    if [[ ! -d "streaming_client_04_80/$EXTRACT_DIR" ]]; then
        print_error "Expected directory not found after extraction: streaming_client_04_80/$EXTRACT_DIR"
        print_info "Listing current directory contents:"
        ls -la
        exit 1
    fi
    
    # Move to the correct location and setup directory structure
    print_info "Setting up directory structure..."
    
    # Remove old directory if it exists
    if [[ -d "$EXTRACT_DIR" ]]; then
        print_warning "Removing existing $EXTRACT_DIR directory"
        rm -rf "$EXTRACT_DIR"
    fi
    
    # Move the extracted directory to the correct location
    mv "streaming_client_04_80/$EXTRACT_DIR" .
    
    # Clean up the temporary extraction directory
    rm -rf "streaming_client_04_80"
    
    print_success "Directory structure setup completed"
}

setup_architecture_libraries() {
    print_info "Setting up architecture-specific libraries..."
    
    cd "$OPERATOR_DIR/$EXTRACT_DIR"
    
    # Check if the architecture directory exists
    if [[ ! -d "lib/$ARCH_DIR" ]]; then
        print_error "Architecture directory not found: lib/$ARCH_DIR"
        print_info "Available directories:"
        ls -la lib/ || echo "No lib directory found"
        exit 1
    fi
    
    # The streaming client needs to keep the architecture-specific directory structure
    # as required by the CMakeLists.txt
    print_info "Architecture libraries are properly structured in lib/$ARCH_DIR/"
    
    # List the available libraries
    print_info "Available libraries for $ARCH_DIR:"
    ls -la "lib/$ARCH_DIR/" || echo "No libraries found"
    
    print_success "Architecture libraries setup completed"
}

cleanup() {
    print_info "Cleaning up temporary files..."
    
    cd "$OPERATOR_DIR"
    
    # Remove NGC download directory
    if [[ -d "$DOWNLOAD_DIR" ]]; then
        rm -rf "$DOWNLOAD_DIR"
        print_info "Removed $DOWNLOAD_DIR"
    fi
    
    print_success "Cleanup completed"
}

verify_setup() {
    print_info "Verifying setup..."
    
    cd "$OPERATOR_DIR"
    
    # Check if the main directory exists
    if [[ ! -d "$EXTRACT_DIR" ]]; then
        print_error "Setup verification failed: $EXTRACT_DIR directory not found"
        exit 1
    fi
    
    # Check if architecture-specific libraries exist
    local lib_count=$(ls -1 "$EXTRACT_DIR/lib/$ARCH_DIR/"*.so* 2>/dev/null | wc -l)
    if [[ $lib_count -eq 0 ]]; then
        print_error "Setup verification failed: No .so libraries found in $EXTRACT_DIR/lib/$ARCH_DIR/"
        exit 1
    fi
    
    # Check if headers exist
    if [[ ! -d "$EXTRACT_DIR/include" ]]; then
        print_warning "Include directory not found: $EXTRACT_DIR/include"
    else
        local header_count=$(ls -1 "$EXTRACT_DIR/include/"*.h 2>/dev/null | wc -l)
        print_info "Found $header_count header files"
    fi
    
    print_success "Setup verification passed"
    print_info "Found $lib_count library files in $EXTRACT_DIR/lib/$ARCH_DIR/"
}

print_summary() {
    echo
    echo -e "${GREEN}"
    echo "=================================================="
    echo "🎉 Streaming Client Setup Completed Successfully!"
    echo "=================================================="
    echo -e "${NC}"
    echo
    echo "📁 Installation location: $OPERATOR_DIR/$EXTRACT_DIR"
    echo "📚 Libraries: $OPERATOR_DIR/$EXTRACT_DIR/lib/$ARCH_DIR/"
    echo "📋 Headers: $OPERATOR_DIR/$EXTRACT_DIR/include/"
    echo
    echo "🚀 Next steps:"
    echo "  1. Build the streaming client operator:"
    echo "     ./holohub build streaming_client_demo_04_80_streaming"
    echo
    echo "  2. Configure your camera in the YAML file:"
    echo "     applications/streaming_client_demo_04_80_streaming/cpp/streaming_client_demo.yaml"
    echo
    echo "  3. Run the streaming client application:"
    echo "     ./holohub run streaming_client_demo_04_80_streaming --language cpp"
    echo
    echo "📋 Camera configuration example:"
    echo "     v4l2_source:"
    echo "       device: \"/dev/video0\""
    echo "       width: 1280"
    echo "       height: 720"
    echo "       frame_rate: 30"
    echo "       pixel_format: \"MJPG\""
    echo
}

# Main execution
main() {
    print_header
    
    # Trap errors and cleanup
    trap 'print_error "Script failed at line $LINENO"' ERR
    
    check_prerequisites
    detect_architecture
    download_client_binaries
    extract_and_setup
    setup_architecture_libraries
    cleanup
    verify_setup
    print_summary
}

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0"
    echo
    echo "This script automatically downloads and sets up the Holoscan streaming client binaries."
    echo
    echo "Prerequisites:"
    echo "  - NGC CLI installed and configured"
    echo "  - Internet connection"
    echo "  - Run from holohub-internal directory or streaming_client_demo application directory"
    echo
    echo "The script will:"
    echo "  1. Download holoscan_client_cloud_streaming from NGC"
    echo "  2. Extract and set up the directory structure"
    echo "  3. Maintain architecture-specific library structure"
    echo "  4. Clean up temporary files"
    echo "  5. Verify the installation"
    echo
    exit 0
fi

# Run main function
main "$@"
