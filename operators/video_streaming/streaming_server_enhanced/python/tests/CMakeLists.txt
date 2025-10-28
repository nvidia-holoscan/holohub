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

# CMakeLists.txt for Streaming Server Enhanced Python Binding Tests

cmake_minimum_required(VERSION 3.20)

if(NOT BUILD_TESTING)
  return()
endif()

# Find Python for pytest execution
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Check if pytest is available
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pytest --version
    RESULT_VARIABLE pytest_available
    OUTPUT_VARIABLE pytest_version
    ERROR_QUIET
)

if(pytest_available EQUAL 0)
    string(STRIP "${pytest_version}" pytest_version)
    message(STATUS "Found pytest: ${pytest_version}")
else()
    message(WARNING "pytest not found at CMake time, but will be checked at test runtime.")
endif()

# Add pytest-based unit tests for Python bindings
# Note: Always add the test, even if pytest check fails at CMake time
# pytest should be available at test runtime
add_test(
    NAME streaming_server_enhanced_pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest 
            -v 
            --tb=short
            ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set test environment with PYTHONPATH
set_tests_properties(streaming_server_enhanced_pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
    TIMEOUT 120
    LABELS "pytest;python;bindings;streaming_server"
)

message(STATUS "StreamingServerOps Python binding tests (pytest) configured.")

