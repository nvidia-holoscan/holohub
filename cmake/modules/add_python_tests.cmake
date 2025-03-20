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

# Function to automatically discover and register pytest tests with CTest
function(add_python_tests)
  # Parse arguments
  set(options "")
  set(oneValueArgs "WORKING_DIRECTORY;INPUT;PREFIX")
  set(multiValueArgs "PYTEST_ARGS")
  cmake_parse_arguments(PARSE_ARGV 0 PYTEST "${options}" "${oneValueArgs}" "${multiValueArgs}")

  # Set default values
  if(NOT PYTEST_INPUT)
    set(PYTEST_INPUT "")
  endif()
  if(NOT PYTEST_WORKING_DIRECTORY)
    set(PYTEST_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  if(NOT PYTEST_PREFIX)
    set(PYTEST_PREFIX "pytest")
  endif()

  # Find Python3 interpreter if not already found
  if(NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  endif()

  # Get the list of tests from pytest - force rootdir to be the working directory
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PYTEST_INPUT} --collect-only -q --rootdir=${PYTEST_WORKING_DIRECTORY}
    WORKING_DIRECTORY ${PYTEST_WORKING_DIRECTORY}
    OUTPUT_VARIABLE pytest_collect_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REPLACE "\n" ";" pytest_list "${pytest_collect_output}")

  # Process each test found
  foreach(test_line ${pytest_list})
    # Extract module file and test name from the full test ID
    # Full pattern might be: path/to/dir/module.py::test_name
    if(test_line MATCHES "(.+)::([a-zA-Z0-9_]+)$")
      set(module_path ${CMAKE_MATCH_1})
      set(test_name ${CMAKE_MATCH_2})

      # Get the module name
      string(REGEX MATCH "([^/]+)$" _ "${module_path}")
      set(module_name ${CMAKE_MATCH_1})
      string(REGEX REPLACE "\\.py$" "" module_name "${module_name}")

      # Add the test with module name in the test name
      add_test(
        NAME "${PYTEST_PREFIX}.${module_name}.${test_name}"
        COMMAND ${Python3_EXECUTABLE} -m pytest "${test_line}" ${PYTEST_PYTEST_ARGS}
        WORKING_DIRECTORY ${PYTEST_WORKING_DIRECTORY}
      )
    endif()
  endforeach()
endfunction()