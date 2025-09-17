#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPERATOR_DIR="${SCRIPT_DIR}/../../operators/streaming_server_enhanced"
NGC_RESOURCE="nvstaging/holoscan/holoscan_server_cloud_streaming:1.0"
DOWNLOAD_DIR="holoscan_server_cloud_streaming_v1.0"
EXTRACT_DIR="holoscan_server_cloud_streaming"

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
    echo "🚀 Holoscan Streaming Server Setup Script"
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
        print_info "Please run this script from the holohub-internal root or streaming_server_demo application directory"
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

download_server_binaries() {
    print_info "Downloading streaming server binaries from NGC..."
    
    cd "$OPERATOR_DIR"
    
    # Clean up any existing downloads
    if [[ -d "$DOWNLOAD_DIR" ]]; then
        print_warning "Removing existing download directory: $DOWNLOAD_DIR"
        rm -rf "$DOWNLOAD_DIR"
    fi
    
    if [[ -f "holoscan_server_cloud_streaming.zip" ]]; then
        print_warning "Removing existing zip file"
        rm -f "holoscan_server_cloud_streaming.zip"
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
    print_info "Extracting and setting up server binaries..."
    
    cd "$OPERATOR_DIR"
    
    # Extract the zip file
    if [[ -f "$DOWNLOAD_DIR/holoscan_server_cloud_streaming.zip" ]]; then
        print_info "Extracting holoscan_server_cloud_streaming.zip..."
        unzip -o "$DOWNLOAD_DIR/holoscan_server_cloud_streaming.zip"
        print_success "Extraction completed"
    else
        print_error "Zip file not found: $DOWNLOAD_DIR/holoscan_server_cloud_streaming.zip"
        exit 1
    fi
    
    # Check if extraction created the expected directory structure
    if [[ ! -d "streaming_server_enhanced/$EXTRACT_DIR" ]]; then
        print_error "Expected directory not found after extraction: streaming_server_enhanced/$EXTRACT_DIR"
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
    mv "streaming_server_enhanced/$EXTRACT_DIR" .
    
    # Clean up the temporary extraction directory
    rm -rf "streaming_server_enhanced"
    
    print_success "Directory structure setup completed"
}

verify_architecture_libraries() {
    print_info "Verifying architecture-specific libraries..."
    
    cd "$OPERATOR_DIR/$EXTRACT_DIR"
    
    # Check if the architecture directory exists
    if [[ ! -d "lib/$ARCH_DIR" ]]; then
        print_error "Architecture directory not found: lib/$ARCH_DIR"
        print_info "Available directories:"
        ls -la lib/ || echo "No lib directory found"
        exit 1
    fi
    
    # The server maintains the original architecture-specific structure
    # No copying needed - just verify the structure is correct
    print_info "Architecture libraries are properly structured in lib/$ARCH_DIR/"
    
    # List the available libraries
    print_info "Available libraries for $ARCH_DIR:"
    ls -la "lib/$ARCH_DIR/" || echo "No libraries found"
    
    print_success "Architecture libraries verification completed"
}

cleanup() {
    print_info "Cleaning up temporary files..."
    
    cd "$OPERATOR_DIR"
    
    # Preserve architecture-specific directories - they are needed for CMakeLists.txt
    # Only remove NGC download directory
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
    echo "🎉 Streaming Server Setup Completed Successfully!"
    echo "=================================================="
    echo -e "${NC}"
    echo
    echo "📁 Installation location: $OPERATOR_DIR/$EXTRACT_DIR"
    echo "📚 Libraries: $OPERATOR_DIR/$EXTRACT_DIR/lib/$ARCH_DIR/"
    echo "📋 Headers: $OPERATOR_DIR/$EXTRACT_DIR/include/"
    echo
    echo "🚀 Next steps:"
    echo "  1. Build the streaming server operator:"
    echo "     ./holohub build streaming_server_demo_04_80_tensor_streaming"
    echo
    echo "  2. Run the streaming server application:"
    echo "     ./holohub run streaming_server_demo_04_80_tensor_streaming --language cpp"
    echo
    echo "  3. Or run with custom resolution:"
    echo "     python applications/streaming_server_demo_04_80_tensor_streaming/python/streaming_server_demo.py --width 1280 --height 720"
    echo
}

# Main execution
main() {
    print_header
    
    # Trap errors and cleanup
    trap 'print_error "Script failed at line $LINENO"' ERR
    
    check_prerequisites
    detect_architecture
    download_server_binaries
    extract_and_setup
    verify_architecture_libraries
    cleanup
    verify_setup
    print_summary
}

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0"
    echo
    echo "This script automatically downloads and sets up the Holoscan streaming server binaries."
    echo
    echo "Prerequisites:"
    echo "  - NGC CLI installed and configured"
    echo "  - Internet connection"
    echo "  - Run from holohub-internal directory or streaming_server_demo application directory"
    echo
    echo "The script will:"
    echo "  1. Download holoscan_server_cloud_streaming from NGC"
    echo "  2. Extract and set up the directory structure" 
    echo "  3. Copy architecture-specific libraries"
    echo "  4. Clean up temporary files"
    echo "  5. Verify the installation"
    echo
    exit 0
fi

# Run main function
main "$@"
