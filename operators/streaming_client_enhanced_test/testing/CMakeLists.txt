# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CMakeLists.txt for Streaming Client Enhanced Operator Python Unit Tests

cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(holoscan 3.2 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find Python for pytest execution
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Check if pytest is available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest; print(pytest.__version__)"
    RESULT_VARIABLE pytest_available
    OUTPUT_VARIABLE pytest_version
    ERROR_QUIET
)

if(pytest_available EQUAL 0)
    string(STRIP "${pytest_version}" pytest_version)
    message(STATUS "Found pytest version: ${pytest_version}")
    
    # Add pytest-based unit tests for Python bindings
    add_test(
        NAME streaming_client_enhanced_python_unit_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest 
                -v 
                --tb=short
                --build-dir=${CMAKE_BINARY_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Set test environment
    set_tests_properties(streaming_client_enhanced_python_unit_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
        TIMEOUT 60
        LABELS "unit;python;bindings"
    )
    
    # Add test for Python bindings specifically
    add_test(
        NAME streaming_client_enhanced_python_bindings_only
        COMMAND ${Python3_EXECUTABLE} -m pytest 
                -v 
                -m "unit and not hardware"
                --build-dir=${CMAKE_BINARY_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_client_op_bindings.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_tests_properties(streaming_client_enhanced_python_bindings_only PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
        TIMEOUT 30
        LABELS "unit;python;bindings;fast"
    )
    
    # Add test for parametrized tests
    add_test(
        NAME streaming_client_enhanced_python_parametrized_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest 
                -v 
                -m "parametrized"
                --build-dir=${CMAKE_BINARY_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_tests_properties(streaming_client_enhanced_python_parametrized_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
        TIMEOUT 45
        LABELS "unit;python;parametrized"
    )
    
else()
    message(WARNING "pytest not found. Python unit tests will be skipped.")
    message(STATUS "To install pytest: pip install pytest")
endif()

# Optional: Add C++ unit tests if GTest is available
find_package(GTest QUIET)
if(GTest_FOUND)
    # Include the parent directory to get access to the operator's sources and headers
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../holoscan_client_cloud_streaming/include)
    
    # Add the C++ unit test executable (if test_streaming_client_op.cpp exists)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_client_op.cpp)
        add_executable(streaming_client_enhanced_cpp_unit_tests test_streaming_client_op.cpp)
        
        # Link against Holoscan SDK, GTest, and the operator's library
        target_link_libraries(streaming_client_enhanced_cpp_unit_tests
                              PRIVATE
                                  holoscan::holoscan
                                  GTest::gtest_main
                                  holoscan::ops::streaming_client
                              )
        
        # Add the C++ test to CTest
        add_test(NAME streaming_client_enhanced_cpp_unit_tests
                 COMMAND $<TARGET_FILE:streaming_client_enhanced_cpp_unit_tests>)
        
        set_tests_properties(streaming_client_enhanced_cpp_unit_tests PROPERTIES
                             PASS_REGULAR_EXPRESSION "^\\[==========\\] \\d+ tests from \\d+ test suites ran\\.$"
                             FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|FAILED"
                             TIMEOUT 30
                             LABELS "unit;cpp;gtest")
    endif()
else()
    message(STATUS "GTest not found. C++ unit tests will be skipped.")
endif()