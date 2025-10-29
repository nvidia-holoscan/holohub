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

# CMake script to test all HoloHub applications
# This script finds all applications with metadata.json and creates tests for each

cmake_minimum_required(VERSION 3.20)
project(HoloHubAllApplicationsTests NONE)

# Get the holohub root directory (passed from CTest script)
if(NOT DEFINED HOLOHUB_ROOT)
  get_filename_component(HOLOHUB_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

message("HoloHub root directory: ${HOLOHUB_ROOT}")

# Option to build or run applications
option(BUILD_ONLY "Only build applications, don't run them" OFF)

if(BUILD_ONLY)
  message("Mode: BUILD ONLY")
  set(HOLOHUB_COMMAND "build")
else()
  message("Mode: RUN")
  set(HOLOHUB_COMMAND "run")
endif()

enable_testing()

# Find all metadata.json files in the applications directory
file(GLOB_RECURSE METADATA_FILES 
  "${HOLOHUB_ROOT}/applications/*/metadata.json")

list(LENGTH METADATA_FILES METADATA_COUNT)
message("\nFound ${METADATA_COUNT} applications with metadata.json")

# Process each metadata.json file
foreach(METADATA_FILE ${METADATA_FILES})
  # Get the directory containing metadata.json
  get_filename_component(METADATA_DIR "${METADATA_FILE}" DIRECTORY)
  get_filename_component(DIR_NAME "${METADATA_DIR}" NAME)
  
  # Skip template directories
  if(DIR_NAME STREQUAL "template" OR DIR_NAME MATCHES "cookiecutter")
    continue()
  endif()
  
  # Check if metadata.json is in a cpp or python subdirectory
  set(LANGUAGE "")
  set(TEST_SUFFIX "")
  
  if(DIR_NAME STREQUAL "cpp")
    set(LANGUAGE "cpp")
    set(TEST_SUFFIX "_cpp")
    # Get the parent directory as the app name
    get_filename_component(METADATA_DIR "${METADATA_DIR}" DIRECTORY)
    get_filename_component(APP_NAME "${METADATA_DIR}" NAME)
  elseif(DIR_NAME STREQUAL "python")
    set(LANGUAGE "python")
    set(TEST_SUFFIX "_python")
    # Get the parent directory as the app name
    get_filename_component(METADATA_DIR "${METADATA_DIR}" DIRECTORY)
    get_filename_component(APP_NAME "${METADATA_DIR}" NAME)
  else()
    # metadata.json is at the application root level
    set(APP_NAME "${DIR_NAME}")
  endif()
  
  # Skip template applications
  if(APP_NAME STREQUAL "template" OR APP_NAME MATCHES "cookiecutter")
    continue()
  endif()
    
  set(TEST_NAME "${APP_NAME}${TEST_SUFFIX}")
  message("  Processing: ${TEST_NAME}")
  
  # Build the command
  if(LANGUAGE STREQUAL "")
    # No language flag
    set(TEST_COMMAND ${HOLOHUB_ROOT}/holohub ${HOLOHUB_COMMAND} ${APP_NAME})
  else()
    # Add language flag
    set(TEST_COMMAND ${HOLOHUB_ROOT}/holohub ${HOLOHUB_COMMAND} ${APP_NAME} --language ${LANGUAGE})
  endif()
  
  # Add test for the application
  message("  Adding test: ${TEST_NAME}")
  add_test(
    NAME ${TEST_NAME}
    COMMAND ${TEST_COMMAND}
    WORKING_DIRECTORY ${HOLOHUB_ROOT}
  )
  
  # Set timeout based on application name
  if(APP_NAME STREQUAL "sam2" OR APP_NAME MATCHES "prohawk")
    set(TEST_TIMEOUT 1000)
  else()
    set(TEST_TIMEOUT 600)
  endif()
  
  set_tests_properties(${TEST_NAME} PROPERTIES
    TIMEOUT ${TEST_TIMEOUT}
    FIXTURES_REQUIRED ${TEST_NAME}_fixture
  )
  
  # Add cleanup test using CTest fixtures
  # This cleanup will ALWAYS run after the test, even if the test fails
  set(CLEANUP_TEST_NAME "${TEST_NAME}_cleanup")
  add_test(
    NAME ${CLEANUP_TEST_NAME}
    COMMAND /bin/bash -c "docker rmi -f holohub:${APP_NAME} 2>/dev/null || true"
  )
  set_tests_properties(${CLEANUP_TEST_NAME} PROPERTIES
    FIXTURES_CLEANUP ${TEST_NAME}_fixture
  )  
endforeach()
