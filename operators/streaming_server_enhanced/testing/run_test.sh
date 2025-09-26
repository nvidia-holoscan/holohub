                                   
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

# Robust test runner for StreamingServer Enhanced testing
# This script provides segfault-resistant execution and comprehensive error handling

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/test_execution.log"
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}
DEFAULT_TIMEOUT=120

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
Usage: $0 <python_script> [test_type] [timeout]

Arguments:
  python_script   Path to Python test script to execute
  test_type      Type of test (functional, unit, infrastructure) [optional]
  timeout        Timeout in seconds (default: $DEFAULT_TIMEOUT) [optional]

Examples:
  $0 video_streaming_server_functional.py
  $0 video_streaming_server_functional.py functional 60
  $0 test_streaming_server_resource.py unit 30

Environment Variables:
  PYTHON_EXECUTABLE   Python executable to use (default: python3)
  STREAMING_SERVER_ENHANCED_TEST_VERBOSE   Enable verbose output (true/false)

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python availability
    if ! command -v "$PYTHON_EXECUTABLE" &> /dev/null; then
        log_error "Python executable not found: $PYTHON_EXECUTABLE"
        return 1
    fi
    
    local python_version
    python_version=$("$PYTHON_EXECUTABLE" --version 2>&1)
    log_info "Using Python: $python_version"
    
    # Check if script exists
    if [[ ! -f "$1" ]]; then
        log_error "Test script not found: $1"
        return 1
    fi
    
    log_info "Prerequisites check passed"
    return 0
}

# Function to setup test environment
setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initialize log file
    {
        echo "================================================================"
        echo "StreamingServer Enhanced Test Execution Log"
        echo "Started at: $(date)"
        echo "Script: $1"
        echo "Test Type: ${2:-auto}"
        echo "Timeout: ${3:-$DEFAULT_TIMEOUT}s"
        echo "================================================================"
    } > "$LOG_FILE"
    
    # Set Python path to include current directory
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    
    # Configure test environment variables
    export STREAMING_SERVER_ENHANCED_TEST_MODE=true
    
    if [[ "${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}" == "true" ]]; then
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=DEBUG
        log_info "Verbose mode enabled"
    else
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=INFO
    fi
    
    log_info "Test environment configured"
}

# Function to execute test with timeout and error handling
execute_test() {
    local script_path="$1"
    local test_type="${2:-auto}"
    local timeout="${3:-$DEFAULT_TIMEOUT}"
    
    log_info "Executing test: $script_path"
    log_info "Test type: $test_type"
    log_info "Timeout: ${timeout}s"
    
    # Prepare command arguments
    local cmd_args=()
    
    # Add minimal flag for infrastructure tests
    if [[ "$test_type" == "infrastructure" ]]; then
        cmd_args+=("--minimal")
        log_info "Running in minimal infrastructure mode"
    fi
    
    # Add verbose flag if enabled
    if [[ "${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    # Execute with timeout and capture output
    local exit_code=0
    local output_file="${SCRIPT_DIR}/test_output_$$.log"
    
    log_info "Starting test execution..."
    
    if timeout "$timeout" "$PYTHON_EXECUTABLE" "$script_path" "${cmd_args[@]}" \
        > "$output_file" 2>&1; then
        exit_code=0
        log_success "Test completed successfully"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "Test timed out after ${timeout} seconds"
        else
            log_error "Test failed with exit code: $exit_code"
        fi
    fi
    
    # Append output to log file
    {
        echo ""
        echo "=== Test Output ==="
        cat "$output_file"
        echo "=== End Test Output ==="
    } >> "$LOG_FILE"
    
    # Display output to console
    cat "$output_file"
    
    # Cleanup temporary output file
    rm -f "$output_file"
    
    return $exit_code
}

# Function to check test success based on output patterns
check_test_success() {
    local script_path="$1"
    local test_type="${2:-auto}"
    
    log_info "Analyzing test results..."
    
    # Define success patterns based on test type
    local success_patterns=()
    local failure_patterns=()
    
    case "$test_type" in
        "functional")
            success_patterns+=(
                "Functional test completed successfully"
                "✅ Functional test completed successfully!"
                "Infrastructure test configured"
            )
            failure_patterns+=(
                "Functional test failed"
                "❌.*test.*failed"
                "ERROR.*test"
            )
            ;;
        "infrastructure")
            success_patterns+=(
                "Infrastructure test configured"
                "Minimal mode enabled"
                "✅.*completed successfully"
            )
            failure_patterns+=(
                "Infrastructure test failed"
                "❌.*failed"
                "ERROR.*infrastructure"
            )
            ;;
        *)
            success_patterns+=(
                "test.*completed successfully"
                "✅.*successful"
                "PASSED"
                "All tests passed"
            )
            failure_patterns+=(
                "test.*failed"
                "❌.*failed"
                "FAILED"
                "ERROR"
                "AssertionError"
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
    
    # Check for failure patterns
    local found_failure=false
    for pattern in "${failure_patterns[@]}"; do
        if grep -q -E "$pattern" "$LOG_FILE"; then
            log_error "Found failure pattern: $pattern"
            found_failure=true
            break
        fi
    done
    
    # Determine overall result
    if [[ "$found_success" == "true" && "$found_failure" == "false" ]]; then
        return 0  # Success
    elif [[ "$found_failure" == "true" ]]; then
        return 1  # Failure
    else
        log_warning "No clear success/failure patterns found"
        return 2  # Unclear
    fi
}

# Function to handle segfaults and crashes
handle_crash() {
    local exit_code=$1
    local script_path="$2"
    
    case $exit_code in
        139)
            log_error "Segmentation fault detected in test: $script_path"
            log_error "This may indicate a memory access error in the C++ components"
            ;;
        134)
            log_error "Abort signal (SIGABRT) detected in test: $script_path"
            log_error "This may indicate an assertion failure or abort() call"
            ;;
        137)
            log_error "Process killed (SIGKILL) - possibly due to memory issues"
            ;;
        *)
            log_error "Test crashed with exit code: $exit_code"
            ;;
    esac
    
    log_info "Collecting crash information..."
    
    # Check for core dumps
    if ls core* &> /dev/null; then
        log_warning "Core dump files found - crash analysis may be possible"
    fi
    
    # Log system information that might be relevant
    log_info "System memory status:"
    free -h >> "$LOG_FILE" 2>&1 || true
    
    log_info "Disk space status:"
    df -h >> "$LOG_FILE" 2>&1 || true
}

# Function to cleanup after test execution
cleanup_test_environment() {
    log_info "Cleaning up test environment..."
    
    # Remove temporary files if any
    find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "test_output_*.log" -delete 2>/dev/null || true
    
    # Unset test-specific environment variables
    unset STREAMING_SERVER_ENHANCED_TEST_MODE
    unset STREAMING_SERVER_ENHANCED_LOG_LEVEL
    
    log_info "Cleanup completed"
}

# Main execution function
main() {
    # Parse arguments
    if [[ $# -lt 1 ]]; then
        log_error "Missing required arguments"
        show_usage
        exit 1
    fi
    
    local script_path="$1"
    local test_type="${2:-auto}"
    local timeout="${3:-$DEFAULT_TIMEOUT}"
    
    # Convert relative path to absolute
    if [[ ! "$script_path" =~ ^/ ]]; then
        script_path="${SCRIPT_DIR}/$script_path"
    fi
    
    log_info "StreamingServer Enhanced Test Runner"
    log_info "Script: $script_path"
    log_info "Test Type: $test_type"
    log_info "Timeout: ${timeout}s"
    
    # Check prerequisites
    if ! check_prerequisites "$script_path"; then
        exit 1
    fi
    
    # Setup test environment
    setup_test_environment "$script_path" "$test_type" "$timeout"
    
    # Execute test
    local exit_code=0
    if execute_test "$script_path" "$test_type" "$timeout"; then
        # Test executed successfully, check results
        if check_test_success "$script_path" "$test_type"; then
            log_success "✅ StreamingServer test PASSED"
            echo "✅ StreamingServer test PASSED"
            exit_code=0
        else
            log_error "❌ StreamingServer test FAILED (result analysis)"
            echo "❌ StreamingServer test FAILED"
            exit_code=1
        fi
    else
        exit_code=$?
        handle_crash $exit_code "$script_path"
        log_error "❌ StreamingServer test FAILED (execution)"
        echo "❌ StreamingServer test FAILED"
    fi
    
    # Cleanup
    cleanup_test_environment
    
    # Final log entry
    {
        echo ""
        echo "================================================================"
        echo "Test execution completed at: $(date)"
        echo "Final result: $([ $exit_code -eq 0 ] && echo "PASSED" || echo "FAILED")"
        echo "Exit code: $exit_code"
        echo "================================================================"
    } >> "$LOG_FILE"
    
    exit $exit_code
}

# Handle script termination signals
trap cleanup_test_environment EXIT
trap 'log_error "Test interrupted by signal"; exit 130' INT TERM

# Run main function with all arguments
main "$@"
