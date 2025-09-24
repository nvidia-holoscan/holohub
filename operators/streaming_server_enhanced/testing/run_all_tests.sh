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

# Comprehensive test runner for StreamingServer Enhanced
# This script runs all test categories: unit, functional, and golden frame tests

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/comprehensive_test.log"
# Simple setup following working client pattern
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}

# Default test configuration
DEFAULT_TIMEOUT=300
DEFAULT_DATA_DIR="/workspace/holohub/data/endoscopy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test results tracking
declare -A test_results
test_count=0
passed_count=0
failed_count=0
skipped_count=0

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

log_section() {
    echo -e "${CYAN}[SECTION]${NC} $*" | tee -a "$LOG_FILE"
}

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  --unit-only          Run only unit tests
  --functional-only    Run only functional tests
  --golden-only        Run only golden frame tests
  --data-dir DIR       Directory containing video data (default: $DEFAULT_DATA_DIR)
  --timeout SECONDS    Test timeout per category (default: $DEFAULT_TIMEOUT)
  --minimal           Run tests in minimal mode
  --verbose           Enable verbose output
  --parallel          Run tests in parallel where possible
  --help              Show this help message

Test Categories:
  unit                 Python unit tests using pytest
  functional           End-to-end functional tests with video pipeline  
  golden               Golden frame visual regression tests

Examples:
  $0                              # Run all test categories
  $0 --unit-only                 # Run only unit tests
  $0 --functional-only --verbose # Run only functional tests with verbose output
  $0 --timeout 600               # Use extended timeout for all tests

Environment Variables:
  PYTHON_EXECUTABLE              Python executable to use (default: python3)
  STREAMING_SERVER_ENHANCED_TEST_VERBOSE  Enable verbose output (true/false)

EOF
}

# Function to initialize test environment
initialize_test_environment() {
    log_info "Initializing comprehensive test environment..."
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initialize log file
    {
        echo "================================================================"
        echo "StreamingServer Enhanced Comprehensive Test Suite"
        echo "Started at: $(date)"
        echo "================================================================"
    } > "$LOG_FILE"
    
    # Set environment variables
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
    export STREAMING_SERVER_ENHANCED_TEST_MODE=comprehensive
    
    if [[ "${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}" == "true" ]]; then
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=DEBUG
        log_info "Verbose mode enabled"
    else
        export STREAMING_SERVER_ENHANCED_LOG_LEVEL=INFO
    fi
    
    log_success "Test environment initialized"
}

# Function to check test prerequisites
check_test_prerequisites() {
    log_info "Basic prerequisite check..."
    
    # Simple Python check (like the working client does)
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        return 1
    fi
    
    log_success "All prerequisites satisfied"
    return 0
}

# Simple pytest execution following the working client pattern
# No complex functions needed - just use direct python3 -m pytest commands

# Function to record test result
record_test_result() {
    local test_name="$1"
    local result="$2"  # PASSED, FAILED, SKIPPED
    local duration="$3"
    
    test_results["$test_name"]="$result:$duration"
    ((test_count++))
    
    case "$result" in
        "PASSED")
            ((passed_count++))
            log_success "✅ $test_name: PASSED (${duration}s)"
            ;;
        "FAILED")
            ((failed_count++))
            log_error "❌ $test_name: FAILED (${duration}s)"
            ;;
        "SKIPPED")
            ((skipped_count++))
            log_warning "⏭️  $test_name: SKIPPED (${duration}s)"
            ;;
    esac
}

# Function to run unit tests
run_unit_tests() {
    local timeout="$1"
    local verbose="$2"
    
    log_section "Running Unit Tests"
    
    local start_time
    start_time=$(date +%s)
    
    # Prepare pytest arguments
    local pytest_args=("-v" "--tb=short" "--maxfail=5")
    
    if [[ "$verbose" == "true" ]]; then
        pytest_args+=("-s" "--log-cli-level=DEBUG")
    fi
    
    # Add timeout
    pytest_args+=("--timeout=$timeout")
    
    # Run specific unit test files
    local unit_test_files=(
        "test_streaming_server_resource.py"
        "test_streaming_server_upstream_op.py" 
        "test_streaming_server_downstream_op.py"
    )
    
    local unit_success=true
    
    for test_file in "${unit_test_files[@]}"; do
        if [[ -f "${SCRIPT_DIR}/$test_file" ]]; then
            log_info "Running unit tests: $test_file"
            
            local file_start_time
            file_start_time=$(date +%s)
            
            # Use simple pytest approach like the working client
            if timeout "$timeout" bash -c "cd '$SCRIPT_DIR' && python3 -m pip install --user pytest --quiet && python3 -m pytest ${pytest_args[*]} $test_file" >> "$LOG_FILE" 2>&1; then
                local duration=$(($(date +%s) - file_start_time))
                record_test_result "Unit Tests: $test_file" "PASSED" "$duration"
            else
                local exit_code=$?
                local duration=$(($(date +%s) - file_start_time))
                
                if [[ $exit_code -eq 124 ]]; then
                    record_test_result "Unit Tests: $test_file" "FAILED" "$duration"
                    log_error "Unit test timed out: $test_file"
                else
                    record_test_result "Unit Tests: $test_file" "FAILED" "$duration"
                    log_error "Unit test failed: $test_file (exit code: $exit_code)"
                fi
                unit_success=false
            fi
        else
            record_test_result "Unit Tests: $test_file" "SKIPPED" "0"
            log_warning "Unit test file not found: $test_file"
        fi
    done
    
    local total_duration=$(($(date +%s) - start_time))
    log_info "Unit tests completed in ${total_duration}s"
    
    return $([ "$unit_success" = true ] && echo 0 || echo 1)
}

# Function to run golden frame tests
run_golden_frame_tests() {
    local timeout="$1"
    local verbose="$2"
    
    log_section "Running Golden Frame Tests"
    
    local start_time
    start_time=$(date +%s)
    
    # Check if golden frame test file exists
    local golden_test_file="${SCRIPT_DIR}/test_golden_frames.py"
    
    if [[ ! -f "$golden_test_file" ]]; then
        record_test_result "Golden Frame Tests" "SKIPPED" "0"
        log_warning "Golden frame test file not found: $golden_test_file"
        return 0
    fi
    
    # Generate golden frames if needed
    local golden_frames_dir="${SCRIPT_DIR}/golden_frames"
    if [[ ! -d "$golden_frames_dir" ]] || [[ -z "$(ls -A "$golden_frames_dir" 2>/dev/null)" ]]; then
        log_info "Generating golden frames..."
        
        local generate_script="${SCRIPT_DIR}/generate_golden_frames.py"
        if [[ -f "$generate_script" ]]; then
            if "$PYTHON_EXECUTABLE" "$generate_script" --output-dir "$golden_frames_dir" \
                --count 10 >> "$LOG_FILE" 2>&1; then
                log_success "Golden frames generated successfully"
            else
                log_error "Failed to generate golden frames"
                record_test_result "Golden Frame Tests" "FAILED" "0"
                return 1
            fi
        else
            log_warning "Golden frame generator not found: $generate_script"
            record_test_result "Golden Frame Tests" "SKIPPED" "0"
            return 0
        fi
    fi
    
    # Run golden frame tests
    local pytest_args=("-v" "--tb=short" "-m" "golden_frame")
    
    if [[ "$verbose" == "true" ]]; then
        pytest_args+=("-s" "--log-cli-level=DEBUG")
    fi
    
    pytest_args+=("--timeout=$timeout")
    
    log_info "Running golden frame tests..."
    
    # Use simple pytest approach like the working client  
    if timeout "$timeout" bash -c "cd '$SCRIPT_DIR' && python3 -m pip install --user pytest --quiet && python3 -m pytest ${pytest_args[*]} $golden_test_file" >> "$LOG_FILE" 2>&1; then
        local duration=$(($(date +%s) - start_time))
        record_test_result "Golden Frame Tests" "PASSED" "$duration"
        return 0
    else
        local exit_code=$?
        local duration=$(($(date +%s) - start_time))
        
        if [[ $exit_code -eq 124 ]]; then
            record_test_result "Golden Frame Tests" "FAILED" "$duration"
            log_error "Golden frame tests timed out"
        else
            record_test_result "Golden Frame Tests" "FAILED" "$duration"
            log_error "Golden frame tests failed (exit code: $exit_code)"
        fi
        return 1
    fi
}

# Function to run functional tests
run_functional_tests() {
    local data_dir="$1"
    local timeout="$2"
    local minimal="$3"
    local verbose="$4"
    
    log_section "Running Functional Tests"
    
    local start_time
    start_time=$(date +%s)
    
    # Check if functional test script exists
    local functional_script="${SCRIPT_DIR}/video_streaming_server_functional.py"
    
    if [[ ! -f "$functional_script" ]]; then
        record_test_result "Functional Tests" "SKIPPED" "0"
        log_warning "Functional test script not found: $functional_script"
        return 0
    fi
    
    # Use the dedicated functional test runner
    local functional_runner="${SCRIPT_DIR}/run_functional_test.sh"
    
    if [[ -f "$functional_runner" ]]; then
        log_info "Using dedicated functional test runner"
        
        local runner_args=("--data-dir" "$data_dir" "--timeout" "$timeout")
        
        if [[ "$minimal" == "true" ]]; then
            runner_args+=("--minimal")
        fi
        
        if [[ "$verbose" == "true" ]]; then
            runner_args+=("--verbose")
        fi
        
        if "$functional_runner" "${runner_args[@]}" >> "$LOG_FILE" 2>&1; then
            local duration=$(($(date +%s) - start_time))
            record_test_result "Functional Tests" "PASSED" "$duration"
            return 0
        else
            local duration=$(($(date +%s) - start_time))
            record_test_result "Functional Tests" "FAILED" "$duration"
            return 1
        fi
    else
        # Fallback to direct script execution
        log_info "Running functional test script directly"
        
        local script_args=("--data-dir" "$data_dir" "--timeout" "$timeout")
        
        if [[ "$minimal" == "true" ]]; then
            script_args+=("--minimal")
        fi
        
        if [[ "$verbose" == "true" ]]; then
            script_args+=("--verbose")
        fi
        
        if timeout "$timeout" "$PYTHON_EXECUTABLE" "$functional_script" \
            "${script_args[@]}" >> "$LOG_FILE" 2>&1; then
            local duration=$(($(date +%s) - start_time))
            record_test_result "Functional Tests" "PASSED" "$duration"
            return 0
        else
            local exit_code=$?
            local duration=$(($(date +%s) - start_time))
            
            if [[ $exit_code -eq 124 ]]; then
                record_test_result "Functional Tests" "FAILED" "$duration"
                log_error "Functional tests timed out"
            else
                record_test_result "Functional Tests" "FAILED" "$duration"
                log_error "Functional tests failed (exit code: $exit_code)"
            fi
            return 1
        fi
    fi
}

# Function to generate comprehensive test report
generate_test_report() {
    local start_time="$1"
    local end_time
    end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    {
        echo ""
        echo "================================================================"
        echo "STREAMING SERVER ENHANCED COMPREHENSIVE TEST REPORT"
        echo "================================================================"
        echo "Test Summary:"
        echo "  Total Tests: $test_count"
        echo "  Passed: $passed_count"
        echo "  Failed: $failed_count"
        echo "  Skipped: $skipped_count"
        echo "  Success Rate: $(awk "BEGIN {printf \"%.1f\", $passed_count/$test_count*100}")%"
        echo "  Total Duration: ${total_duration}s"
        echo ""
        echo "Detailed Results:"
        
        for test_name in "${!test_results[@]}"; do
            local result_info="${test_results[$test_name]}"
            local result="${result_info%%:*}"
            local duration="${result_info##*:}"
            
            case "$result" in
                "PASSED") echo "  ✅ $test_name (${duration}s)" ;;
                "FAILED") echo "  ❌ $test_name (${duration}s)" ;;
                "SKIPPED") echo "  ⏭️  $test_name (${duration}s)" ;;
            esac
        done
        
        echo ""
        echo "================================================================"
        echo "Completed at: $(date)"
        echo "================================================================"
    } | tee -a "$LOG_FILE"
}

# Function to cleanup test environment
cleanup_test_environment() {
    log_info "Cleaning up test environment..."
    
    # Remove temporary files
    find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$SCRIPT_DIR" -name "*_output_*.log" -delete 2>/dev/null || true
    
    # Unset environment variables
    unset STREAMING_SERVER_ENHANCED_TEST_MODE
    unset STREAMING_SERVER_ENHANCED_LOG_LEVEL
    
    log_info "Cleanup completed"
}

# Main execution function
main() {
    local run_unit="true"
    local run_functional="true"
    local run_golden="true"
    local data_dir="$DEFAULT_DATA_DIR"
    local timeout="$DEFAULT_TIMEOUT"
    local minimal="false"
    local verbose="${STREAMING_SERVER_ENHANCED_TEST_VERBOSE:-false}"
    local parallel="false"
    
    local start_time
    start_time=$(date +%s)
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit-only)
                run_functional="false"
                run_golden="false"
                shift
                ;;
            --functional-only)
                run_unit="false"
                run_golden="false"
                shift
                ;;
            --golden-only)
                run_unit="false"
                run_functional="false"
                shift
                ;;
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
            --parallel)
                parallel="true"
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
    
    log_info "StreamingServer Enhanced Comprehensive Test Suite"
    log_info "Unit Tests: $run_unit"
    log_info "Functional Tests: $run_functional" 
    log_info "Golden Frame Tests: $run_golden"
    log_info "Data Directory: $data_dir"
    log_info "Timeout per category: ${timeout}s"
    log_info "Minimal Mode: $minimal"
    log_info "Verbose: $verbose"
    log_info "Parallel: $parallel"
    
    # Initialize test environment
    initialize_test_environment
    
    # Check prerequisites
    if ! check_test_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # Run tests based on configuration
    local overall_success=true
    
    if [[ "$run_unit" == "true" ]]; then
        if ! run_unit_tests "$timeout" "$verbose"; then
            overall_success=false
        fi
    fi
    
    if [[ "$run_golden" == "true" ]]; then
        if ! run_golden_frame_tests "$timeout" "$verbose"; then
            overall_success=false
        fi
    fi
    
    if [[ "$run_functional" == "true" ]]; then
        if ! run_functional_tests "$data_dir" "$timeout" "$minimal" "$verbose"; then
            overall_success=false
        fi
    fi
    
    # Generate final report
    generate_test_report "$start_time"
    
    # Cleanup
    cleanup_test_environment
    
    # Final result
    if [[ "$overall_success" == "true" ]]; then
        log_success "✅ All StreamingServer tests PASSED"
        echo "✅ All StreamingServer tests PASSED"
        exit 0
    else
        log_error "❌ Some StreamingServer tests FAILED"
        echo "❌ Some StreamingServer tests FAILED"
        exit 1
    fi
}

# Handle script termination signals
trap cleanup_test_environment EXIT
trap 'log_error "Test suite interrupted by signal"; exit 130' INT TERM

# Run main function with all arguments
main "$@"
