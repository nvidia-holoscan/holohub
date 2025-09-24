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

# CMakeLists.txt for Streaming Client Enhanced Operator Unit Tests

cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(holoscan 3.2 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# Copy test files to build directory (only copy files that exist)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/conftest.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_client_op_bindings.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_streaming_client_op.cpp" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/pytest.ini" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config.yaml" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Copy enhanced testing files
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_golden_frames.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/generate_golden_frames.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_client_functional.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Copy shell script wrappers
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/run_functional_test.sh" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/run_cpp_test.sh" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.sh" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Copy golden frames if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/golden_frames")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/golden_frames" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Register individual pytest tests with CTest using add_python_tests (only if pytest is available)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest"
    RESULT_VARIABLE pytest_available
    ERROR_QUIET
)

if(pytest_available EQUAL 0)
    include(add_python_tests)
    add_python_tests(
        INPUT "."
        PYTEST_ARGS "-v" "-s" "--tb=short" "--log-cli-level=INFO"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Set PYTHONPATH for all pytest tests to match the working Test #1 environment
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        if (_variableName MATCHES "^_PYTEST_.*")
            unset(${_variableName})
        endif()
    endforeach()
    
    # Get list of all tests and add PYTHONPATH environment to pytest tests
    get_property(all_tests DIRECTORY PROPERTY TESTS)
    foreach(test_name ${all_tests})
        if(test_name MATCHES "^pytest\\.")
            set_tests_properties("${test_name}" PROPERTIES
                ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                TIMEOUT 600
                LABELS "unit;operator;python;bindings"
            )
        endif()
    endforeach()
    
    message(STATUS "Added pytest tests to CTest for streaming_client_enhanced")
    message(STATUS "  - Python binding tests: test_streaming_client_op_bindings.py")
    message(STATUS "  - PYTHONPATH configured: ${CMAKE_BINARY_DIR}/python/lib")
else()
    message(WARNING "pytest not available - skipping automatic test discovery. Install pytest to enable individual test registration.")
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
        message(STATUS "Added C++ unit tests to CTest for streaming_client_enhanced")
    endif()
else()
    message(STATUS "GTest not found. C++ unit tests will be skipped.")
endif()

# Add shell script-based tests for robust testing
add_test(NAME streaming_client_enhanced_functional_test
         COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/run_functional_test.sh" 
                 "${CMAKE_BINARY_DIR}" 
                 "${CMAKE_CURRENT_BINARY_DIR}/video_streaming_client_functional.py"
                 ""
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set_tests_properties(streaming_client_enhanced_functional_test PROPERTIES
                     PASS_REGULAR_EXPRESSION "✅.*test PASSED"
                     FAIL_REGULAR_EXPRESSION "❌.*test FAILED|FATAL|CRITICAL|Segmentation fault"
                     TIMEOUT 120
                     LABELS "functional;python;integration")

# Add golden frame generation test
add_test(NAME streaming_client_enhanced_golden_frame_generation
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/generate_golden_frames.py" 
                 --frames 5 --config
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set_tests_properties(streaming_client_enhanced_golden_frame_generation PROPERTIES
                     PASS_REGULAR_EXPRESSION "✅ Golden frame generation complete"
                     TIMEOUT 30
                     LABELS "golden;generation;unit")

# Add comprehensive test suite
add_test(NAME streaming_client_enhanced_comprehensive_test_suite
         COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/run_all_tests.sh" --timeout 60
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set_tests_properties(streaming_client_enhanced_comprehensive_test_suite PROPERTIES
                     PASS_REGULAR_EXPRESSION "🎉 All tests passed"
                     FAIL_REGULAR_EXPRESSION "❌.*failed|Some tests failed"
                     TIMEOUT 300
                     LABELS "comprehensive;integration;all")

# Add infrastructure test with robust wrapper
add_test(NAME streaming_client_enhanced_infrastructure_test
         COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/run_test.sh"
                 "${CMAKE_CURRENT_BINARY_DIR}/video_streaming_client_functional.py"
                 "infrastructure"
                 "60"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

set_tests_properties(streaming_client_enhanced_infrastructure_test PROPERTIES
                     PASS_REGULAR_EXPRESSION "✅.*test PASSED"
                     FAIL_REGULAR_EXPRESSION "❌.*test FAILED"
                     TIMEOUT 80
                     LABELS "infrastructure;python;robust")

message(STATUS "Added enhanced test suite to CTest for streaming_client_enhanced")
message(STATUS "  - Golden frame generation test")
message(STATUS "  - Functional test with video pipeline")
message(STATUS "  - Infrastructure test with robust wrapper")
message(STATUS "  - Comprehensive test suite runner")