#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Comprehensive test suite runner for streaming_client_enhanced
# Combines both original robust testing and enhanced modern testing approaches

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Test configuration
TIMEOUT_DEFAULT=60
VERBOSE=false
SKIP_UNIT=false
SKIP_FUNCTIONAL=false
SKIP_GOLDEN=false
BUILD_DIR=""

# Test results tracking
declare -a TEST_RESULTS=()
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Help function
show_help() {
    echo "StreamingClient Enhanced Comprehensive Test Suite"
    echo "================================================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -b, --build-dir DIR     Specify build directory"
    echo "  -t, --timeout SECONDS   Set timeout for individual tests (default: $TIMEOUT_DEFAULT)"
    echo "  --skip-unit             Skip unit tests"
    echo "  --skip-functional       Skip functional tests"
    echo "  --skip-golden           Skip golden frame tests"
    echo "  --unit-only             Run only unit tests"
    echo "  --functional-only       Run only functional tests"
    echo "  --golden-only           Run only golden frame tests"
    echo
    echo "EXAMPLES:"
    echo "  $0                      # Run all tests"
    echo "  $0 --unit-only          # Run only unit tests"
    echo "  $0 --skip-golden        # Run all tests except golden frame tests"
    echo "  $0 -v -t 120            # Verbose mode with 2-minute timeout"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT_DEFAULT="$2"
            shift 2
            ;;
        --skip-unit)
            SKIP_UNIT=true
            shift
            ;;
        --skip-functional)
            SKIP_FUNCTIONAL=true
            shift
            ;;
        --skip-golden)
            SKIP_GOLDEN=true
            shift
            ;;
        --unit-only)
            SKIP_FUNCTIONAL=true
            SKIP_GOLDEN=true
            shift
            ;;
        --functional-only)
            SKIP_UNIT=true
            SKIP_GOLDEN=true
            shift
            ;;
        --golden-only)
            SKIP_UNIT=true
            SKIP_FUNCTIONAL=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Utility functions
log_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

# Function to record test result
record_test_result() {
    local test_name="$1"
    local result="$2"
    local duration="$3"
    local details="$4"
    
    TEST_RESULTS+=("$test_name|$result|$duration|$details")
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$result" = "PASSED" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Function to run a single test with timeout and error handling
run_single_test() {
    local test_name="$1"
    local test_command="$2"
    local timeout="${3:-$TIMEOUT_DEFAULT}"
    
    echo
    log_header "🧪 Running: $test_name"
    
    if [ "$VERBOSE" = true ]; then
        log_info "Command: $test_command"
        log_info "Timeout: ${timeout}s"
    fi
    
    local start_time=$(date +%s)
    local result="FAILED"
    local details=""
    
    # Create temporary log file for this test
    local temp_log=$(mktemp)
    
    if timeout "$timeout" bash -c "$test_command" > "$temp_log" 2>&1; then
        result="PASSED"
        details="Test completed successfully"
        log_success "$test_name completed successfully"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            details="Test timed out after ${timeout}s"
            log_warning "$test_name timed out after ${timeout}s"
        else
            details="Test failed with exit code $exit_code"
            log_error "$test_name failed with exit code $exit_code"
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Show test output if verbose or if test failed
    if [ "$VERBOSE" = true ] || [ "$result" = "FAILED" ]; then
        echo "--- Test Output ---"
        cat "$temp_log"
        echo "--- End Output ---"
    fi
    
    # Clean up
    rm -f "$temp_log"
    
    record_test_result "$test_name" "$result" "${duration}s" "$details"
    
    return $( [ "$result" = "PASSED" ] && echo 0 || echo 1 )
}

# Auto-detect build directory if not provided
detect_build_directory() {
    if [ -n "$BUILD_DIR" ] && [ -d "$BUILD_DIR" ]; then
        return
    fi
    
    local search_paths=(
        "/workspace/holohub/build-streaming_client_demo_enhanced"
        "/workspace/holohub/build"
        "../../../build-streaming_client_demo_enhanced"
        "../build"
        "./build"
    )
    
    for path in "${search_paths[@]}"; do
        if [ -d "$path" ]; then
            BUILD_DIR="$path"
            log_info "Auto-detected build directory: $BUILD_DIR"
            return
        fi
    done
    
    log_warning "Build directory not found. Some tests may be skipped."
}

# Main test execution
main() {
    log_header "🚀 StreamingClient Enhanced Comprehensive Test Suite"
    
    echo "Test Configuration:"
    echo "  📁 Script directory: $SCRIPT_DIR"
    echo "  ⏱️ Default timeout: ${TIMEOUT_DEFAULT}s"
    echo "  🔊 Verbose mode: $VERBOSE"
    echo "  🧪 Skip unit tests: $SKIP_UNIT"
    echo "  🎬 Skip functional tests: $SKIP_FUNCTIONAL"
    echo "  🖼️ Skip golden frame tests: $SKIP_GOLDEN"
    
    detect_build_directory
    echo "  🏗️ Build directory: ${BUILD_DIR:-'Not found'}"
    
    # Test 1: Python Unit Tests (pytest-based)
    if [ "$SKIP_UNIT" = false ]; then
        log_header "📚 Unit Testing Phase"
        
        # Check if pytest is available
        if command -v pytest >/dev/null 2>&1 || python3 -m pytest --version >/dev/null 2>&1; then
            # Run Python binding tests
            run_single_test \
                "Python Bindings Unit Tests" \
                "cd '$SCRIPT_DIR' && python3 -m pytest test_streaming_client_op_bindings.py -v" \
                $TIMEOUT_DEFAULT
            
            # Run golden frame tests if not skipped
            if [ "$SKIP_GOLDEN" = false ]; then
                run_single_test \
                    "Golden Frame Unit Tests" \
                    "cd '$SCRIPT_DIR' && python3 -m pytest test_golden_frames.py -v -m unit" \
                    $TIMEOUT_DEFAULT
            fi
        else
            log_warning "pytest not available, installing on-the-fly..."
            run_single_test \
                "Python Bindings Unit Tests (with pip install)" \
                "cd '$SCRIPT_DIR' && python3 -m pip install --user pytest --quiet && python3 -m pytest test_streaming_client_op_bindings.py -v" \
                $((TIMEOUT_DEFAULT + 30))
        fi
        
        # Run C++ unit tests if available
        if [ -n "$BUILD_DIR" ] && [ -f "$BUILD_DIR/operators/streaming_client_enhanced/streaming_client_enhanced_cpp_unit_tests" ]; then
            run_single_test \
                "C++ Unit Tests (GTest)" \
                "$BUILD_DIR/operators/streaming_client_enhanced/streaming_client_enhanced_cpp_unit_tests" \
                $TIMEOUT_DEFAULT
        fi
    fi
    
    # Test 2: Golden Frame Testing
    if [ "$SKIP_GOLDEN" = false ]; then
        log_header "🖼️ Golden Frame Testing Phase"
        
        # Generate golden frames if they don't exist
        if [ ! -d "$SCRIPT_DIR/golden_frames" ]; then
            run_single_test \
                "Golden Frame Generation" \
                "cd '$SCRIPT_DIR' && python3 generate_golden_frames.py --frames 10 --config" \
                $TIMEOUT_DEFAULT
        fi
        
        # Run golden frame validation tests
        if command -v pytest >/dev/null 2>&1; then
            run_single_test \
                "Golden Frame Integration Tests" \
                "cd '$SCRIPT_DIR' && python3 -m pytest test_golden_frames.py -v -m integration" \
                $TIMEOUT_DEFAULT
        fi
    fi
    
    # Test 3: Functional Testing with Video Pipeline
    if [ "$SKIP_FUNCTIONAL" = false ]; then
        log_header "🎬 Functional Testing Phase"
        
        # Run Python functional test
        run_single_test \
            "Python Functional Test" \
            "cd '$SCRIPT_DIR' && python3 video_streaming_client_functional.py --frames 20 --verbose" \
            $((TIMEOUT_DEFAULT * 2))
        
        # Run functional test with wrapper script
        if [ -f "$SCRIPT_DIR/run_functional_test.sh" ]; then
            run_single_test \
                "Functional Test with Wrapper" \
                "'$SCRIPT_DIR/run_functional_test.sh' '$BUILD_DIR' '$SCRIPT_DIR/video_streaming_client_functional.py' ''" \
                $((TIMEOUT_DEFAULT * 2))
        fi
        
        # Run C++ functional test if available
        if [ -n "$BUILD_DIR" ] && [ -f "$BUILD_DIR/applications/streaming_client_demo_enhanced/cpp/streaming_client_demo_enhanced" ]; then
            local cpp_config="$BUILD_DIR/applications/streaming_client_demo_enhanced/cpp/streaming_client_demo.yaml"
            if [ ! -f "$cpp_config" ]; then
                cpp_config="$SCRIPT_DIR/unit_test_config.yaml"
            fi
            
            run_single_test \
                "C++ Application Functional Test" \
                "'$SCRIPT_DIR/run_cpp_test.sh' '$BUILD_DIR/applications/streaming_client_demo_enhanced/cpp/streaming_client_demo_enhanced' '$cpp_config' ''" \
                $TIMEOUT_DEFAULT
        fi
    fi
    
    # Test 4: Integration Testing (if both unit and functional are enabled)
    if [ "$SKIP_UNIT" = false ] && [ "$SKIP_FUNCTIONAL" = false ]; then
        log_header "🔗 Integration Testing Phase"
        
        # Run infrastructure test with robust wrapper
        if [ -f "$SCRIPT_DIR/run_test.sh" ]; then
            run_single_test \
                "Infrastructure Test with Robust Wrapper" \
                "'$SCRIPT_DIR/run_test.sh' '$SCRIPT_DIR/video_streaming_client_functional.py' 'infrastructure' $TIMEOUT_DEFAULT" \
                $((TIMEOUT_DEFAULT + 10))
        fi
        
        # Test timeout handling
        run_single_test \
            "Timeout Handling Test" \
            "cd '$SCRIPT_DIR' && timeout 5 python3 -c 'import time; time.sleep(10)' || echo 'Timeout handled correctly'" \
            10
    fi
}

# Print final results
print_final_results() {
    echo
    log_header "📊 Test Results Summary"
    
    echo "Overall Statistics:"
    echo "  🧪 Total tests: $TOTAL_TESTS"
    echo "  ✅ Passed: $PASSED_TESTS"
    echo "  ❌ Failed: $FAILED_TESTS"
    echo "  📈 Success rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo
    
    echo "Individual Test Results:"
    printf "%-40s %-10s %-10s %s\n" "Test Name" "Result" "Duration" "Details"
    printf "%-40s %-10s %-10s %s\n" "$(printf '=%.0s' {1..40})" "$(printf '=%.0s' {1..10})" "$(printf '=%.0s' {1..10})" "$(printf '=%.0s' {1..20})"
    
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r name status duration details <<< "$result"
        
        if [ "$status" = "PASSED" ]; then
            printf "%-40s ${GREEN}%-10s${NC} %-10s %s\n" "$name" "$status" "$duration" "$details"
        else
            printf "%-40s ${RED}%-10s${NC} %-10s %s\n" "$name" "$status" "$duration" "$details"
        fi
    done
    
    echo
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "🎉 All tests passed! StreamingClient Enhanced is working correctly."
        return 0
    else
        log_error "Some tests failed. Check the detailed results above."
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo
    log_info "Cleaning up temporary files..."
    # Remove any temporary files created during testing
    find "$SCRIPT_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$SCRIPT_DIR/test_outputs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
}

# Main execution with error handling
trap cleanup EXIT

main
exit_code=$?

print_final_results
final_exit_code=$?

# Exit with appropriate code
if [ $exit_code -eq 0 ] && [ $final_exit_code -eq 0 ]; then
    exit 0
else
    exit 1
fi
