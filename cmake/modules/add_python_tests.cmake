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
    COMMAND ${Python3_EXECUTABLE} -m pytest ${PYTEST_INPUT} --collect-only --rootdir=${PYTEST_WORKING_DIRECTORY}
    WORKING_DIRECTORY ${PYTEST_WORKING_DIRECTORY}
    OUTPUT_VARIABLE pytest_collect_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE pytest_collect_error
    RESULT_VARIABLE pytest_collect_result
  )
  set(PYTEST_SEARCH_DIR "${PYTEST_WORKING_DIRECTORY}/${PYTEST_INPUT}")
  if(NOT pytest_collect_result EQUAL 0)
    message(FATAL_ERROR "Error collecting pytest tests in ${PYTEST_SEARCH_DIR} (returned ${pytest_collect_result}):\n${pytest_collect_error}")
  endif()

  if(NOT pytest_collect_output)
    message(WARNING "No pytest tests found in ${PYTEST_SEARCH_DIR}")
    return()
  endif()

  message(DEBUG "pytest_collect_output: ${pytest_collect_output}")
  string(REPLACE "\n" ";" pytest_list "${pytest_collect_output}")

  # Process each test found
  set(tests_added FALSE)
  foreach(test_line ${pytest_list})
    message(DEBUG "pytest collected test: ${test_line}")
    # Extract module file and test name from the full test ID
    # Support both regular tests (path/to/module.py::test_name)
    # and parameterized tests (path/to/module.py::test_name[param1-param2])
    if(test_line MATCHES "(.+)::([^\\[]+)(\\[.+\\])?$")
      set(module_path ${CMAKE_MATCH_1})
      set(test_name ${CMAKE_MATCH_2})
      set(test_params ${CMAKE_MATCH_3})
      message(DEBUG "module_path: ${module_path}")
      message(DEBUG "test_name: ${test_name}")
      message(DEBUG "test_params: ${test_params}")

      # --- Construct the ctest name ---
      string(REGEX MATCH "([^/]+)$" _ "${module_path}")
      set(module_name ${CMAKE_MATCH_1})
      string(REGEX REPLACE "\\.py$" "" module_name "${module_name}")
      if(test_params)
        string(REPLACE "[" "" test_params "${test_params}")
        string(REPLACE "]" "" test_params "${test_params}")
        set(ctest_name "${PYTEST_PREFIX}.${module_name}.${test_name}_${test_params}")
      else()
        set(ctest_name "${PYTEST_PREFIX}.${module_name}.${test_name}")
      endif()

      # --- Construct the command ---
      # Separate input pytest args with spaces
      string(JOIN " " PYTEST_ARGS_STR ${PYTEST_PYTEST_ARGS})
      # Configure color output if tty is available (--color=auto does not work with ctest wrapping)
      set(PYTEST_COLOR_ARG "--color=\$(tty -s && echo yes || echo no)")
      string(PREPEND PYTEST_ARGS_STR "${PYTEST_COLOR_ARG} ")
      # Create the pytest command, ensuring usage of the requested python interpreter
      set(PYTEST_CMD "${Python3_EXECUTABLE} -m pytest ${test_line} ${PYTEST_ARGS_STR}")

      # --- Wrap the individual pytest with a ctest ---
      # The command is wrapped in bash to run the command substitution in PYTEST_COLOR_ARG
      message(STATUS "Adding CTest: ${ctest_name}")
      add_test(
        NAME "${ctest_name}"
        COMMAND bash -c "${PYTEST_CMD}"
        WORKING_DIRECTORY ${PYTEST_WORKING_DIRECTORY}
      )
      set(tests_added TRUE)
    endif()
  endforeach()

  if(NOT tests_added)
    message(WARNING "No pytest tests were found in ${PYTEST_SEARCH_DIR}")
  endif()
endfunction()