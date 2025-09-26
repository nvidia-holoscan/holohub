#!/bin/bash
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

# Functional test runner for StreamingServer Enhanced
# This script handles video data discovery, fallback modes, and robust functional testing

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/functional_test.log"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}
DEFAULT_TIMEOUT=120
DEFAULT_DATA_DIR="/workspace/holohub/data/endoscopy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  --data-dir DIR       Directory containing video data (default: $DEFAULT_DATA_DIR)
  --timeout SECONDS    Test timeout in seconds (default: $DEFAULT_TIMEOUT)
  --minimal           Run minimal infrastructure test only
  --verbose           Enable verbose output
  --help              Show this help message

Examples:
  $0                                    # Run with defaults
  $0 --data-dir /path/to/video/data    # Use custom data directory
  $0 --minimal                         # Run minimal infrastructure test
  $0 --timeout 180 --verbose           # Extended timeout with verbose output

Environment Variables:
  PYTHON_EXECUTABLE                    Python executable to use (default: python3)
  STREAMING_SERVER_ENHANCED_TEST_VERBOSE   Enable verbose output (true/false)

EOF
}

# Function to check video data availability
check_video_data() {
    local data_dir="$1"
    
    log_info "Checking video data availability in: $data_dir"
    
    if [[ ! -d "$data_dir" ]]; then
        log_warning "Data directory not found: $data_dir"
        return 1
    fi
    
    # Look for video files
    local video_files=()
    while IFS= read -r -d $'\0' file; do
        video_files+=("$file")
    done < <(find "$data_dir" -type f \( -name "*.264" -o -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) -print0 2>/dev/null)
    
    if [[ ${#video_files[@]} -eq 0 ]]; then
        log_warning "No video files found in $data_dir"
        log_info "Supported formats: .264, .mp4, .avi, .mov"
        return 1
    fi
    
    log_success "Found ${#video_files[@]} video files:"
    for file in "${video_files[@]}"; do
        local size
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "unknown")
        log_info "  - $(basename "$file") ($size)"
    done
    
    return 0
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python availability
    if ! command -v "$PYTHON_EXECUTABLE" &> /dev/null; then
        log_error "Python executable not found: $PYTHON_EXECUTABLE"
        return 1
    fi
    
    local python_version
    python_version=$("$PYTHON_EXECUTABLE" --version 2>&1)
    log_info "Python version: $python_version"
    
    # Check for required Python packages
    local required_packages=("numpy" "pytest")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! "$PYTHON_EXECUTABLE" -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_warning "Missing Python packages: ${missing_packages[*]}"
        log_info "Consider installing with: pip install ${missing_packages[*]}"
    fi
    
    # Check available memory
    local memory_mb
    if memory_mb=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null); then
        log_info "Available memory: ${memory_mb}MB"
        if [[ $memory_mb -lt 1024 ]]; then
            log_warning "Low memory available: ${memory_mb}MB (recommended: 1GB+)"
        fi
    fi
    
    # Check disk space
    local disk_space
    if disk_space=$(df "$SCRIPT_DIR" 2>/dev/null | awk 'NR==2 {print $4}'); then
        local disk_space_mb=$((disk_space / 1024))
        log_info "Available disk space: ${disk_space_mb}MB"
        if [[ $disk_space_mb -lt 512 ]]; then
            log_warning "Low disk space: ${disk_space_mb}MB (recommended: 512MB+)"
        fi
    fi
    
    log_success "System requirements check completed"
    return 0
}

# Function to determine test mode
determine_test_mode() {
    local data_dir="$1"
    local force_minimal="$2"
    
    if [[ "$force_minimal" == "true" ]]; then
        log_info "Minimal mode requested by user" >&2
        echo "minimal"
        return 0
    fi
    
    if check_video_data "$data_dir"; then
        log_info "Video data available - will attempt full functional test" >&2
        echo "functional"
    else
        log_warning "Video data not available - falling back to infrastructure test" >&2
        echo "infrastructure"
    fi
}

# Function to setup test environment
setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initialize log file
    {
        echo "================================================================"
        echo "StreamingServer Enhanced Functional Test Log"
        echo "Started at: $(date)"
        echo "================================================================"
    } > "$LOG_FILE"
    
    # Set environment variables
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    export STREAMING_SERVER_ENHANCED_TEST_MODE=functional
    
    if [[ "${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}" == "true" ]]; then
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=DEBUG
        log_info "Verbose mode enabled"
    else
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=INFO
    fi
    
    log_success "Test environment configured"
}

# Function to run functional test
run_functional_test() {
    local data_dir="$1"
    local test_mode="$2"
    local timeout="$3"
    local verbose="$4"
    
    log_info "Running StreamingServer functional test"
    log_info "Mode: $test_mode"
    log_info "Data directory: $data_dir"
    log_info "Timeout: ${timeout}s"
    
    # Prepare test script path
    local test_script="${SCRIPT_DIR}/video_streaming_server_functional.py"
    
    if [[ ! -f "$test_script" ]]; then
        log_error "Functional test script not found: $test_script"
        return 1
    fi
    
    # Prepare command arguments
    local cmd_args=("--data-dir" "$data_dir" "--timeout" "$timeout")
    
    if [[ "$test_mode" == "minimal" || "$test_mode" == "infrastructure" ]]; then
        cmd_args+=("--minimal")
        log_info "Running in minimal infrastructure mode"
    fi
    
    if [[ "$verbose" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    # Execute test with timeout
    local exit_code=0
    local output_file="${SCRIPT_DIR}/functional_output_$$.log"
    
    log_info "Executing functional test..."
    log_info "Command: $PYTHON_EXECUTABLE $test_script ${cmd_args[*]}"
    
    if timeout "$timeout" "$PYTHON_EXECUTABLE" "$test_script" "${cmd_args[@]}" \
        > "$output_file" 2>&1; then
        exit_code=0
        log_success "Functional test execution completed"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "Functional test timed out after ${timeout} seconds"
        else
            log_error "Functional test failed with exit code: $exit_code"
        fi
    fi
    
    # Append output to log and display
    {
        echo ""
        echo "=== Functional Test Output ==="
        cat "$output_file"
        echo "=== End Functional Test Output ==="
    } >> "$LOG_FILE"
    
    cat "$output_file"
    
    # Cleanup
    rm -f "$output_file"
    
    return $exit_code
}

# Function to validate test results
validate_test_results() {
    local test_mode="$1"
    
    log_info "Validating test results..."
    
    # Define success patterns based on test mode
    local success_patterns=()
    
    case "$test_mode" in
        "functional")
            success_patterns+=(
                "Functional test completed successfully"
                "✅ Functional test completed successfully!"
                "✅.*successful"
            )
            ;;
        "minimal"|"infrastructure")
            success_patterns+=(
                "Infrastructure test configured"
                "Minimal mode enabled"
                "✅ Functional test completed successfully!"
                "Infrastructure test configured \\(minimal mode\\)"
            )
            ;;
    esac
    
    # Check for success patterns
    local found_success=false
    for pattern in "${success_patterns[@]}"; do
        if grep -q -E "$pattern" "$LOG_FILE"; then
            log_success "Found success pattern: $pattern"
            found_success=true
            break
        fi
    done
    
    # Check for common failure patterns
    local failure_patterns=(
        "❌.*failed"
        "ERROR.*test"
        "FAILED"
        "Test.*failed"
        "AssertionError"
        "Exception.*occurred"
    )
    
    local found_failure=false
    for pattern in "${failure_patterns[@]}"; do
        if grep -q -E "$pattern" "$LOG_FILE"; then
            log_error "Found failure pattern: $pattern"
            found_failure=true
            break
        fi
    done
    
    # Return result
    if [[ "$found_success" == "true" && "$found_failure" == "false" ]]; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

# Function to cleanup test environment
cleanup_test_environment() {
    log_info "Cleaning up test environment..."
    
    # Remove temporary files
    find "$SCRIPT_DIR" -name "functional_output_*.log" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    
    # Unset environment variables
    unset STREAMING_SERVER_ENHANCED_TEST_MODE
    unset STREAMING_SERVER_ENHANCED_LOG_LEVEL
    
    log_info "Cleanup completed"
}

# Function to generate test summary
generate_test_summary() {
    local test_mode="$1"
    local exit_code="$2"
    local start_time="$3"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    {
        echo ""
        echo "================================================================"
        echo "STREAMING SERVER ENHANCED FUNCTIONAL TEST SUMMARY"
        echo "================================================================"
        echo "Test Mode: $test_mode"
        echo "Duration: ${duration}s"
        echo "Result: $([ $exit_code -eq 0 ] && echo "PASSED" || echo "FAILED")"
        echo "Completed at: $(date)"
        echo "================================================================"
    } | tee -a "$LOG_FILE"
}

# Main execution function
main() {
    local data_dir="$DEFAULT_DATA_DIR"
    local timeout="$DEFAULT_TIMEOUT"
    local minimal="false"
    local verbose="${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}"
    local start_time
    start_time=$(date +%s)
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data-dir)
                data_dir="$2"
                shift 2
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            --minimal)
                minimal="true"
                shift
                ;;
            --verbose)
                verbose="true"
                export STREAMING_SERVER_ENHANCED_TEST_VERBOSE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "StreamingServer Enhanced Functional Test Runner"
    log_info "Data directory: $data_dir"
    log_info "Timeout: ${timeout}s"
    log_info "Minimal mode: $minimal"
    log_info "Verbose: $verbose"
    
    # Setup test environment
    setup_test_environment
    
    # Check system requirements
    if ! check_system_requirements; then
        log_error "System requirements check failed"
        exit 1
    fi
    
    # Determine test mode
    local test_mode
    test_mode=$(determine_test_mode "$data_dir" "$minimal")
    log_info "Test mode determined: $test_mode"
    
    # Run functional test
    local exit_code=0
    if run_functional_test "$data_dir" "$test_mode" "$timeout" "$verbose"; then
        # Validate results
        if validate_test_results "$test_mode"; then
            log_success "✅ StreamingServer functional test PASSED"
            echo "✅ StreamingServer functional test PASSED"
            exit_code=0
        else
            log_error "❌ StreamingServer functional test FAILED (validation)"
            echo "❌ StreamingServer functional test FAILED"
            exit_code=1
        fi
    else
        exit_code=$?
        log_error "❌ StreamingServer functional test FAILED (execution)"
        echo "❌ StreamingServer functional test FAILED"
    fi
    
    # Generate summary
    generate_test_summary "$test_mode" "$exit_code" "$start_time"
    
    # Cleanup
    cleanup_test_environment
    
    exit $exit_code
}

# Handle script termination signals
trap cleanup_test_environment EXIT
trap 'log_error "Functional test interrupted by signal"; exit 130' INT TERM

# Run main function with all arguments
main "$@"
