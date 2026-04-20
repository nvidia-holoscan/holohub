# SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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
project(atracsys_visualizer_apps LANGUAGES NONE)

option(HOLOHUB_DOWNLOAD_DATASETS "Prepare application datasets" ON)
option(ATRACSYS_VISUALIZER_ENABLE_LIVE_CAMERA
       "Build atracsys_visualizer with the optional atracsys_camera operator path"
       OFF)
set(ATRACSYS_VISUALIZER_SOURCE_RECORDINGS_DIR "" CACHE PATH
    "Optional path to local Atracsys recordings used to trim a development replay sample")

if(ATRACSYS_VISUALIZER_ENABLE_LIVE_CAMERA)
  set(OP_atracsys_camera ON CACHE BOOL "Build the atracsys_camera operator")
else()
  set(OP_atracsys_camera OFF CACHE BOOL "Build the atracsys_camera operator")
endif()

if(NOT DEFINED HOLOHUB_BUILD_OPERATORS)
  set(HOLOHUB_BUILD_OPERATORS "" CACHE STRING "Operators to build dynamically")
endif()

if(ATRACSYS_VISUALIZER_ENABLE_LIVE_CAMERA)
  if(NOT "atracsys_camera" IN_LIST HOLOHUB_BUILD_OPERATORS)
    set(HOLOHUB_BUILD_OPERATORS "${HOLOHUB_BUILD_OPERATORS};atracsys_camera" CACHE STRING "Operators to build dynamically" FORCE)
  endif()
else()
  if("atracsys_camera" IN_LIST HOLOHUB_BUILD_OPERATORS)
    list(REMOVE_ITEM HOLOHUB_BUILD_OPERATORS "atracsys_camera")
    set(HOLOHUB_BUILD_OPERATORS "${HOLOHUB_BUILD_OPERATORS}" CACHE STRING "Operators to build dynamically" FORCE)
  endif()
endif()

if(NOT DEFINED HOLOHUB_DATA_DIR OR HOLOHUB_DATA_DIR STREQUAL "")
  message(FATAL_ERROR "HOLOHUB_DATA_DIR must be set when building with datasets")
endif()

if(HOLOHUB_DOWNLOAD_DATASETS AND NOT EXISTS "${HOLOHUB_DATA_DIR}/atracsys_visualizer/ir_base.gxf_entities")
  set(ATRACSYS_VISUALIZER_DATASET_URL
      "https://github.com/waylandio/holohub/releases/download/dataset-v1.0/atracsys_visualizer.zip")
  set(ATRACSYS_VISUALIZER_DATASET_SHA256 "674551dee1c40bf4f2b62c0ea23be01b6126d0cf549971d87b2fd86b0d4895e9")
  message(STATUS "Downloading Atracsys Visualizer demo dataset...")
  file(DOWNLOAD "${ATRACSYS_VISUALIZER_DATASET_URL}"
       "${CMAKE_CURRENT_BINARY_DIR}/atracsys_visualizer.zip"
       EXPECTED_HASH SHA256=${ATRACSYS_VISUALIZER_DATASET_SHA256}
       TLS_VERIFY ON
       SHOW_PROGRESS
       STATUS DOWNLOAD_STATUS)

  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download dataset. Status: ${DOWNLOAD_STATUS}")
  endif()

  message(STATUS "Extracting Atracsys Visualizer demo dataset...")
  file(MAKE_DIRECTORY "${HOLOHUB_DATA_DIR}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_BINARY_DIR}/atracsys_visualizer.zip"
    WORKING_DIRECTORY "${HOLOHUB_DATA_DIR}"
    RESULT_VARIABLE EXTRACT_STATUS
  )
  if(NOT EXTRACT_STATUS EQUAL 0)
    message(FATAL_ERROR "Failed to extract dataset.")
  endif()
  file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/atracsys_visualizer.zip")
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()
