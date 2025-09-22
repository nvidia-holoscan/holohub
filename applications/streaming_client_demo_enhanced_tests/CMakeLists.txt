# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(streaming_client_demo_enhanced_tests LANGUAGES NONE)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
endif()

# Add C++ subdirectory
add_subdirectory(cpp)

# Add Python subdirectory
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Add operator pytest tests when BUILD_TESTING is enabled
if(BUILD_TESTING)
  # Import operator pytest tests
  set(OPERATOR_TESTS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/operators/streaming_client_enhanced_test/testing")
  set(OPERATOR_TESTS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/operator_tests")
  
  # Copy operator pytest files to application's test directory
  if(EXISTS "${OPERATOR_TESTS_SOURCE_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/conftest.py" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/test_streaming_client_op_bindings.py" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/test_utils.py" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/pytest.ini" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/requirements.txt" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")
    file(COPY "${OPERATOR_TESTS_SOURCE_DIR}/unit_test_config.yaml" DESTINATION "${OPERATOR_TESTS_BINARY_DIR}")

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
      message(STATUS "Added operator pytest tests to application test suite")
    else()
      message(STATUS "pytest not available - skipping operator pytest tests. Install pytest to enable.")
    endif()
  else()
    message(STATUS "Operator tests not found at ${OPERATOR_TESTS_SOURCE_DIR}")
  endif()
endif()