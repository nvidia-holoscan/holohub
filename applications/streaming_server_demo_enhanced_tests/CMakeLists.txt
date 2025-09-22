# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
project(streaming_server_demo_enhanced_tests)

# Add C++ and Python subdirectories
message(STATUS "=== PROCESSING PARENT streaming_server_demo_enhanced/CMakeLists.txt ===")

# Add C++ subdirectory
add_subdirectory(cpp)

# Add Python subdirectory
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Add operator pytest tests when BUILD_TESTING is enabled
if(BUILD_TESTING)
  # Import operator pytest tests for streaming_server_enhanced_test
  set(OPERATOR_TESTS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/operators/streaming_server_enhanced_test/testing")
  set(OPERATOR_TESTS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/operator_tests")
  
  # Copy operator pytest files to application's test directory
  if(EXISTS "${OPERATOR_TESTS_SOURCE_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")

    # Register operator pytest tests with CTest (only if pytest is available)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE pytest_available
        ERROR_QUIET
    )
    
    if(pytest_available EQUAL 0)
      include(add_python_tests)
      add_python_tests(
        INPUT "operator_tests"
        PYTEST_ARGS "-v" "-s" "--tb=short" "--log-cli-level=INFO" "--build-dir=${CMAKE_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
      message(STATUS "Added streaming_server_enhanced_test operator pytest tests to application test suite")
    else()
      message(STATUS "pytest not available - skipping operator pytest tests. Install pytest to enable.")
    endif()
  else()
    message(STATUS "Streaming server operator tests not found at ${OPERATOR_TESTS_SOURCE_DIR}")
  endif()
endif()

# Copy Python files to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/streaming_server_demo.py
    ${CMAKE_BINARY_DIR}/streaming_server_demo.py
    COPYONLY
)

# Create run_python.sh script
file(WRITE ${CMAKE_BINARY_DIR}/run_python.sh "#!/bin/bash\n\npython3 streaming_server_demo.py \"$@\"\n")
file(CHMOD ${CMAKE_BINARY_DIR}/run_python.sh
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "=== COMPLETED PARENT streaming_server_demo_enhanced/CMakeLists.txt ===")