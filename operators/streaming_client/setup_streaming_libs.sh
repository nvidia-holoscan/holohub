#!/bin/bash

# =============================================================================
# NVIDIA Holoscan Streaming Client Library Setup Script
# =============================================================================
# This script automates the download and setup of streaming client libraries
# for the streaming_client operator in HoloHub.
#
# Usage: ./setup_streaming_libs.sh [--force] [--help]
#   --force: Force re-download even if libraries already exist
#   --help:  Show this help message
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGC_RESOURCE="nvidia/holoscan_client_cloud_streaming:0.1"
ZIP_FILE="holoscan_client_cloud_streaming_v0.1/holoscan_client_cloud_streaming.zip"
EXTRACT_DIR="holoscan_client_cloud_streaming_v0.1"
LIB_DIR="${SCRIPT_DIR}/lib"
INCLUDE_DIR="${SCRIPT_DIR}/include"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
NVIDIA Holoscan Streaming Client Library Setup Script

This script automates the download and setup of streaming client libraries
for the streaming_client operator in HoloHub.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --force     Force re-download even if libraries already exist
    --help      Show this help message

DESCRIPTION:
    This script will:
    1. Detect your system architecture (x86_64 or aarch64)
    2. Download NVIDIA streaming libraries from NGC
    3. Extract and organize libraries in the correct directories
    4. Clean up temporary files
    5. Verify the installation

REQUIREMENTS:
    - NGC CLI tool must be installed and configured
    - Internet connection for downloading
    - Write permissions in the operator directory

EXAMPLES:
    ./setup_streaming_libs.sh          # Normal setup
    ./setup_streaming_libs.sh --force  # Force re-download

EOF
}

# Function to check if NGC CLI is available
check_ngc_cli() {
    if ! command -v ngc &> /dev/null; then
        print_error "NGC CLI is not installed or not in PATH"
        print_error "Please install NGC CLI: https://docs.nvidia.com/ngc/ngc-cli/install-guide/index.html"
        exit 1
    fi
    print_success "NGC CLI found"
}

# Function to detect system architecture
detect_architecture() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            print_error "Supported architectures: x86_64, aarch64"
            exit 1
            ;;
    esac
}

# Function to check if libraries already exist
check_existing_libs() {
    if [[ -d "$LIB_DIR" ]] && [[ -d "$INCLUDE_DIR" ]] && [[ "$(ls -A "$LIB_DIR")" ]]; then
        return 0  # Libraries exist
    else
        return 1  # Libraries don't exist
    fi
}

# Function to backup existing libraries
backup_existing_libs() {
    if [[ -d "$LIB_DIR" ]]; then
        local backup_dir="${LIB_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        print_status "Backing up existing lib directory to: $backup_dir"
        mv "$LIB_DIR" "$backup_dir"
    fi
    
    if [[ -d "$INCLUDE_DIR" ]]; then
        local backup_dir="${INCLUDE_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        print_status "Backing up existing include directory to: $backup_dir"
        mv "$INCLUDE_DIR" "$backup_dir"
    fi
}

# Function to download NGC resource
download_ngc_resource() {
    print_status "Downloading NGC resource: $NGC_RESOURCE"
    
    # Remove existing download directory if it exists
    if [[ -d "$EXTRACT_DIR" ]]; then
        print_status "Removing existing download directory..."
        rm -rf "$EXTRACT_DIR"
    fi
    
    # Download the resource
    if ngc registry resource download-version "$NGC_RESOURCE" --dest .; then
        print_success "NGC resource downloaded successfully"
    else
        print_error "Failed to download NGC resource"
        exit 1
    fi
}

# Function to extract libraries
extract_libraries() {
    local arch="$1"
    
    print_status "Extracting libraries for architecture: $arch"
    
    # Check if zip file exists
    if [[ ! -f "$ZIP_FILE" ]]; then
        print_error "Zip file not found: $ZIP_FILE"
        exit 1
    fi
    
    # Extract the zip file
    if unzip -o "$ZIP_FILE"; then
        print_success "Libraries extracted successfully"
    else
        print_error "Failed to extract libraries"
        exit 1
    fi
    
    # Create directories
    mkdir -p "$LIB_DIR" "$INCLUDE_DIR"
    
    # Copy architecture-specific libraries
    local arch_lib_dir="lib/$arch"
    if [[ -d "$arch_lib_dir" ]]; then
        print_status "Copying $arch libraries..."
        cp "$arch_lib_dir"/*.so* "$LIB_DIR"/ 2>/dev/null || print_warning "No .so files found in $arch_lib_dir"
        cp "$arch_lib_dir"/* "$LIB_DIR"/ 2>/dev/null || print_warning "No files found in $arch_lib_dir"
        
        # Copy plugins if they exist
        if [[ -d "$arch_lib_dir/plugins" ]]; then
            print_status "Copying plugins..."
            cp -r "$arch_lib_dir/plugins" "$LIB_DIR"/
        fi
    else
        print_error "Architecture-specific library directory not found: $arch_lib_dir"
        exit 1
    fi
    
    # Copy include files if they exist
    if [[ -d "include" ]]; then
        print_status "Copying include files..."
        cp -r include/* "$INCLUDE_DIR"/
    else
        print_warning "No include directory found in extracted files"
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    print_status "Cleaning up temporary files..."
    
    # Remove architecture-specific directories
    rm -rf lib/x86_64 lib/aarch64
    
    # Remove NGC download directory
    if [[ -d "$EXTRACT_DIR" ]]; then
        rm -rf "$EXTRACT_DIR"
    fi
    
    print_success "Cleanup completed"
}

# Function to verify installation
verify_installation() {
    local arch="$1"
    
    print_status "Verifying installation..."
    
    # Check if lib directory exists and has files
    if [[ ! -d "$LIB_DIR" ]] || [[ ! "$(ls -A "$LIB_DIR")" ]]; then
        print_error "Library directory is empty or missing"
        exit 1
    fi
    
    # Count library files
    local lib_count=$(find "$LIB_DIR" -name "*.so*" | wc -l)
    print_success "Found $lib_count library files"
    
    # Check for key libraries
    local key_libs=("libStreamingClient.so" "libNvStreamBase.so" "libStreamClientShared.so")
    for lib in "${key_libs[@]}"; do
        if find "$LIB_DIR" -name "$lib*" | grep -q .; then
            print_success "Key library found: $lib"
        else
            print_warning "Key library not found: $lib"
        fi
    done
    
    # Check include directory
    if [[ -d "$INCLUDE_DIR" ]] && [[ "$(ls -A "$INCLUDE_DIR")" ]]; then
        local header_count=$(find "$INCLUDE_DIR" -name "*.h" -o -name "*.hpp" | wc -l)
        print_success "Found $header_count header files"
        
        # Check for key headers
        local key_headers=("StreamingClient.h" "VideoFrame.h")
        for header in "${key_headers[@]}"; do
            if find "$INCLUDE_DIR" -name "$header" | grep -q .; then
                print_success "Key header found: $header"
            else
                print_warning "Key header not found: $header"
            fi
        done
    else
        print_warning "Include directory is empty or missing"
    fi
    
    print_success "Installation verification completed"
}

# Function to show installation summary
show_summary() {
    local arch="$1"
    
    cat << EOF

${GREEN}=== INSTALLATION SUMMARY ===${NC}
Architecture:     $arch
Library Directory: $LIB_DIR
Include Directory: $INCLUDE_DIR
Library Files:    $(find "$LIB_DIR" -name "*.so*" 2>/dev/null | wc -l)
Header Files:     $(find "$INCLUDE_DIR" -name "*.h" -o -name "*.hpp" 2>/dev/null | wc -l)

${GREEN}Expected Libraries:${NC}
- libStreamingClient.so
- libStreamClientShared.so
- libNvStreamBase.so
- libNvStreamingSession.so
- libNvStreamServer.so
- libPoco.so
- libcrypto.so.3
- libssl.so.3
- libcudart.so.12

${GREEN}Expected Headers:${NC}
- StreamingClient.h
- VideoFrame.h

${GREEN}=== NEXT STEPS ===${NC}
1. The streaming client libraries are now ready to use
2. You can build the streaming_client operator with CMake
3. Libraries are automatically detected by the CMake configuration

${BLUE}=== TROUBLESHOOTING ===${NC}
- If build errors occur, verify all .so files are in: $LIB_DIR
- If header errors occur, verify headers are in: $INCLUDE_DIR
- Run this script with --force to re-download if issues persist
- Ensure library permissions are correct (644)

EOF
}

# Main function
main() {
    local force_download=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                force_download=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_error "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    echo -e "${BLUE}=== NVIDIA Holoscan Streaming Client Library Setup ===${NC}"
    echo ""
    
    # Check prerequisites
    check_ngc_cli
    
    # Detect architecture
    local arch=$(detect_architecture)
    print_success "Detected architecture: $arch"
    
    # Check if libraries already exist
    if check_existing_libs && [[ "$force_download" != true ]]; then
        print_warning "Libraries already exist in $LIB_DIR"
        print_warning "Use --force to re-download or remove the directories manually"
        
        # Still show summary of existing installation
        show_summary "$arch"
        exit 0
    fi
    
    # Backup existing libraries if force download
    if [[ "$force_download" == true ]]; then
        backup_existing_libs
    fi
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Download and setup libraries
    download_ngc_resource
    extract_libraries "$arch"
    cleanup_temp_files
    verify_installation "$arch"
    
    # Show installation summary
    show_summary "$arch"
    
    print_success "Streaming client library setup completed successfully!"
}

# Run main function with all arguments
main "$@"
