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

# StreamingServer Enhanced Testing Integration
# This file integrates the comprehensive testing suite with CMake/CTest

cmake_minimum_required(VERSION 3.20)

if(NOT BUILD_TESTING)
    message(STATUS "Testing disabled, skipping StreamingServer Enhanced tests")
    return()
endif()

# Enable testing
include(CTest)
enable_testing()

# Get current binary directory for test execution
set(STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Configuring StreamingServer Enhanced testing suite")
message(STATUS "Test binary directory: ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}")

# Find required packages for C++ tests
find_package(GTest QUIET)
find_package(GMock QUIET)
find_package(holoscan REQUIRED CONFIG)

# Find Python executable
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Find pytest
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m pytest --version
    RESULT_VARIABLE PYTEST_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(PYTEST_RESULT EQUAL 0)
    set(PYTEST_EXECUTABLE ${PYTHON_EXECUTABLE} -m pytest)
    message(STATUS "Found pytest: ${PYTEST_EXECUTABLE}")
else()
    message(WARNING "pytest not found - some tests may not run properly")
    set(PYTEST_EXECUTABLE ${PYTHON_EXECUTABLE} -m pytest)
endif()

# Copy test files to binary directory
message(STATUS "Copying test files to binary directory")

# Core test files
set(TEST_FILES
    conftest.py
    pytest.ini
    mock_holoscan_framework.py
    test_utils.py
    test_streaming_server_resource.py
    test_streaming_server_upstream_op.py
    test_streaming_server_downstream_op.py
    test_streaming_server_resource_bindings.py
    test_streaming_server_upstream_op_bindings.py
    test_streaming_server_downstream_op_bindings.py
    test_golden_frames.py
    video_streaming_server_functional.py
    generate_golden_frames.py
)

# Shell script wrappers
set(SHELL_SCRIPTS
    run_test.sh
    run_functional_test.sh
    run_all_tests.sh
)

# Copy test files
foreach(test_file ${TEST_FILES})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}
        ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/${test_file}
        COPYONLY
    )
endforeach()

# Copy and make shell scripts executable (force fresh copy every configure)
foreach(script ${SHELL_SCRIPTS})
    # Remove any existing cached version first, then copy fresh
    file(REMOVE ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/${script})
    
    # Copy fresh version with executable permissions
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${script}
         DESTINATION ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}
         FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
                        
    message(STATUS "Copied fresh shell script: ${script}")
endforeach()

# Create golden frames directory
file(MAKE_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/golden_frames)

# Copy any existing golden frames
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/golden_frames)
    file(GLOB GOLDEN_FRAME_FILES ${CMAKE_CURRENT_SOURCE_DIR}/golden_frames/*)
    if(GOLDEN_FRAME_FILES)
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/golden_frames/
             DESTINATION ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/golden_frames)
        message(STATUS "Copied existing golden frames")
    endif()
endif()

# Configure test environment variables
set(TEST_ENV_VARS
    "PYTHONPATH=${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}"
    "STREAMING_SERVER_ENHANCED_TEST_MODE=cmake"
    "STREAMING_SERVER_ENHANCED_TEST_DATA_DIR=/workspace/holohub/data/endoscopy"
)

#####################################################################
# C++ Unit Tests (GTest)
#####################################################################

if(GTest_FOUND AND GMock_FOUND)
    message(STATUS "Configuring C++ unit tests")
    
    # C++ test utilities library
    add_library(streaming_server_enhanced_test_utils
        test_utilities.cpp
    )
    
    target_include_directories(streaming_server_enhanced_test_utils PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/operators/streaming_server_enhanced
    )
    
    target_link_libraries(streaming_server_enhanced_test_utils
        PUBLIC
        holoscan::core
        GTest::gtest
        GTest::gmock
    )
    
    # StreamingServerResource C++ tests
    add_executable(test_streaming_server_resource_cpp
        test_streaming_server_resource_cpp.cpp
    )
    
    target_link_libraries(test_streaming_server_resource_cpp
        streaming_server_enhanced_test_utils
        holoscan::ops::streaming_server_enhanced
        GTest::gtest_main
        GTest::gmock
    )
    
    # StreamingServerUpstreamOp C++ tests
    add_executable(test_streaming_server_upstream_op_cpp
        test_streaming_server_upstream_op_cpp.cpp
    )
    
    target_link_libraries(test_streaming_server_upstream_op_cpp
        streaming_server_enhanced_test_utils
        holoscan::ops::streaming_server_enhanced
        GTest::gtest_main
        GTest::gmock
    )
    
    # StreamingServerDownstreamOp C++ tests
    add_executable(test_streaming_server_downstream_op_cpp
        test_streaming_server_downstream_op_cpp.cpp
    )
    
    target_link_libraries(test_streaming_server_downstream_op_cpp
        streaming_server_enhanced_test_utils
        holoscan::ops::streaming_server_enhanced
        GTest::gtest_main
        GTest::gmock
    )
    
    # Register C++ tests with CTest
    add_test(NAME streaming_server_enhanced_cpp_resource_tests
             COMMAND test_streaming_server_resource_cpp
             WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})
    
    set_tests_properties(streaming_server_enhanced_cpp_resource_tests PROPERTIES
                         TIMEOUT 60
                         LABELS "unit;cpp;streaming_server;resource")
    
    # Enable automatic pytest discovery like streaming_client_enhanced
    include(add_python_tests)
    add_python_tests(
        INPUT "."
        PYTEST_ARGS "-v" "-s" "--tb=short" "--log-cli-level=INFO"
        WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}
    )
    message(STATUS "Added automatic pytest discovery for streaming_server_enhanced")
    message(STATUS "  - Python binding tests will be auto-discovered")
    
    add_test(NAME streaming_server_enhanced_cpp_upstream_tests
             COMMAND test_streaming_server_upstream_op_cpp
             WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})
    
    set_tests_properties(streaming_server_enhanced_cpp_upstream_tests PROPERTIES
                         TIMEOUT 60
                         LABELS "unit;cpp;streaming_server;upstream")
    
    add_test(NAME streaming_server_enhanced_cpp_downstream_tests
             COMMAND test_streaming_server_downstream_op_cpp
             WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})
    
    set_tests_properties(streaming_server_enhanced_cpp_downstream_tests PROPERTIES
                         TIMEOUT 60
                         LABELS "unit;cpp;streaming_server;downstream")
    
    message(STATUS "C++ unit tests configured (3 test suites)")
else()
    message(STATUS "GTest/GMock not found - skipping C++ unit tests")
endif()

#####################################################################
# Python Unit Tests
#####################################################################

message(STATUS "Configuring Python unit tests")

# Test 1: StreamingServerResource unit tests
add_test(NAME streaming_server_enhanced_resource_unit_tests
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short 
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_resource.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_resource_unit_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "unit;python;streaming_server;resource")

# Test 2: StreamingServerUpstreamOp unit tests  
add_test(NAME streaming_server_enhanced_upstream_unit_tests
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_upstream_op.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_upstream_unit_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "unit;python;streaming_server;upstream")

# Test 3: StreamingServerDownstreamOp unit tests
add_test(NAME streaming_server_enhanced_downstream_unit_tests  
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_downstream_op.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_downstream_unit_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "unit;python;streaming_server;downstream")

# Test 4: StreamingServerResource Python bindings tests
add_test(NAME streaming_server_enhanced_resource_bindings_tests
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short 
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_resource_bindings.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_resource_bindings_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "bindings;python;streaming_server;resource")

# Test 5: StreamingServerUpstreamOp Python bindings tests  
add_test(NAME streaming_server_enhanced_upstream_bindings_tests
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_upstream_op_bindings.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_upstream_bindings_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "bindings;python;streaming_server;upstream")

# Test 6: StreamingServerDownstreamOp Python bindings tests
add_test(NAME streaming_server_enhanced_downstream_bindings_tests  
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_streaming_server_downstream_op_bindings.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_downstream_bindings_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 120
                     LABELS "bindings;python;streaming_server;downstream")

#####################################################################
# Golden Frame Tests
#####################################################################

message(STATUS "Configuring golden frame tests")

# Test 7: Golden frame generation
add_test(NAME streaming_server_enhanced_golden_frame_generation
         COMMAND ${PYTHON_EXECUTABLE} 
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/generate_golden_frames.py
                 --output-dir ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/golden_frames
                 --count 10
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_golden_frame_generation PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 60
                     LABELS "golden_frame;generation;streaming_server")

# Test 8: Golden frame validation tests
add_test(NAME streaming_server_enhanced_golden_frame_tests
         COMMAND ${PYTEST_EXECUTABLE} -v --tb=short -m golden_frame
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_golden_frames.py
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_golden_frame_tests PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 180
                     LABELS "golden_frame;validation;streaming_server"
                     DEPENDS streaming_server_enhanced_golden_frame_generation)

#####################################################################
# Functional Tests
#####################################################################

message(STATUS "Configuring functional tests")

# Test 9: Functional test with robust wrapper
add_test(NAME streaming_server_enhanced_functional_test
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_functional_test.sh
                 --data-dir /workspace/holohub/data/endoscopy
                 --timeout 120
                 --minimal
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_functional_test PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 150
                     LABELS "functional;streaming_server;pipeline")

# Test 10: Infrastructure test (minimal mode)
add_test(NAME streaming_server_enhanced_infrastructure_test
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_test.sh
                 ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/video_streaming_server_functional.py
                 infrastructure
                 60
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_infrastructure_test PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 80
                     LABELS "infrastructure;streaming_server;minimal")

#####################################################################
# Comprehensive Test Suite
#####################################################################

message(STATUS "Configuring comprehensive test suite")

# Test 11: Comprehensive test suite runner
add_test(NAME streaming_server_enhanced_comprehensive_test_suite
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_all_tests.sh
                 --timeout 120
                 --minimal
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_comprehensive_test_suite PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 600
                     LABELS "comprehensive;streaming_server;full_suite")

#####################################################################
# Individual Test Categories
#####################################################################

# Test 12: Unit tests only (via comprehensive runner)
add_test(NAME streaming_server_enhanced_unit_only_suite
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_all_tests.sh
                 --unit-only
                 --timeout 120
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_unit_only_suite PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 200
                     LABELS "unit;streaming_server;suite")

# Test 10: Golden frame tests only (via comprehensive runner)  
add_test(NAME streaming_server_enhanced_golden_only_suite
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_all_tests.sh
                 --golden-only
                 --timeout 120
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_golden_only_suite PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 200
                     LABELS "golden_frame;streaming_server;suite")

# Test 11: Functional tests only (via comprehensive runner)
add_test(NAME streaming_server_enhanced_functional_only_suite
         COMMAND bash ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/run_all_tests.sh
                 --functional-only
                 --minimal
                 --timeout 120
         WORKING_DIRECTORY ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR})

set_tests_properties(streaming_server_enhanced_functional_only_suite PROPERTIES
                     ENVIRONMENT "${TEST_ENV_VARS}"
                     TIMEOUT 200
                     LABELS "functional;streaming_server;suite")

#####################################################################
# Test Dependencies and Ordering
#####################################################################

# Set up test dependencies for logical ordering
set_tests_properties(streaming_server_enhanced_golden_frame_tests PROPERTIES
                     DEPENDS streaming_server_enhanced_golden_frame_generation)

#####################################################################
# Custom Test Targets
#####################################################################

# Create custom target for quick unit testing
add_custom_target(streaming_server_enhanced_test_quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "unit" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running quick StreamingServer Enhanced unit tests"
)

# Create custom target for full testing
add_custom_target(streaming_server_enhanced_test_full
    COMMAND ${CMAKE_CTEST_COMMAND} -L "streaming_server" --output-on-failure  
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running full StreamingServer Enhanced test suite"
)

# Create custom target for functional testing only
add_custom_target(streaming_server_enhanced_test_functional
    COMMAND ${CMAKE_CTEST_COMMAND} -L "functional" --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running StreamingServer Enhanced functional tests"
)

#####################################################################
# Test Configuration Summary
#####################################################################

message(STATUS "StreamingServer Enhanced test configuration complete:")
if(GTest_FOUND AND GMock_FOUND)
    message(STATUS "  - C++ unit tests: 3 test suites")
    message(STATUS "  - Python unit tests: 3 test suites")
    message(STATUS "  - Python binding tests: 3 test suites")
    message(STATUS "  - Golden frame tests: 2 test suites") 
    message(STATUS "  - Functional tests: 2 test suites")
    message(STATUS "  - Comprehensive tests: 4 test suites")
    message(STATUS "  - Total tests: 17 (C++ + Python + Bindings)")
else()
    message(STATUS "  - Python unit tests: 3 test suites")
    message(STATUS "  - Python binding tests: 3 test suites")
    message(STATUS "  - Golden frame tests: 2 test suites") 
    message(STATUS "  - Functional tests: 2 test suites")
    message(STATUS "  - Comprehensive tests: 4 test suites")
    message(STATUS "  - Total tests: 14 (Python + Bindings)")
endif()
message(STATUS "  - Custom targets: streaming_server_enhanced_test_quick, streaming_server_enhanced_test_full, streaming_server_enhanced_test_functional")

# Create a summary file for reference
file(WRITE ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_summary.txt
"StreamingServer Enhanced Test Suite Summary
============================================

Test Categories:
1. Unit Tests (3 suites)
   - streaming_server_enhanced_resource_unit_tests
   - streaming_server_enhanced_upstream_unit_tests  
   - streaming_server_enhanced_downstream_unit_tests

2. Golden Frame Tests (2 suites)
   - streaming_server_enhanced_golden_frame_generation
   - streaming_server_enhanced_golden_frame_tests

3. Functional Tests (2 suites)
   - streaming_server_enhanced_functional_test
   - streaming_server_enhanced_infrastructure_test

4. Comprehensive Tests (4 suites)
   - streaming_server_enhanced_comprehensive_test_suite
   - streaming_server_enhanced_unit_only_suite
   - streaming_server_enhanced_golden_only_suite
   - streaming_server_enhanced_functional_only_suite

Usage:
  ctest -L streaming_server                    # Run all StreamingServer tests
  ctest -L unit                                # Run unit tests only
  ctest -L functional                          # Run functional tests only
  ctest -L golden_frame                        # Run golden frame tests only
  
  make streaming_server_enhanced_test_quick    # Quick unit tests
  make streaming_server_enhanced_test_full     # Full test suite
  make streaming_server_enhanced_test_functional # Functional tests only

Test Files Location: ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}
")

message(STATUS "Test summary written to: ${STREAMING_SERVER_ENHANCED_TEST_BINARY_DIR}/test_summary.txt")
