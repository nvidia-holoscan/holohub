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

if(DEFINED HOLOHUB_BUILD_OPERATORS)
  string(REPLACE "\"" "" holohub_build_operators "${HOLOHUB_BUILD_OPERATORS}")
else()
  set(holohub_build_operators "")
endif()

if(ATRACSYS_VISUALIZER_ENABLE_LIVE_CAMERA)
  if(NOT "atracsys_camera" IN_LIST holohub_build_operators)
    list(APPEND holohub_build_operators "atracsys_camera")
  endif()
  set(OP_atracsys_camera ON CACHE BOOL "Build the atracsys_camera operator" FORCE)
else()
  list(REMOVE_ITEM holohub_build_operators "atracsys_camera")
  set(OP_atracsys_camera OFF CACHE BOOL "Build the atracsys_camera operator" FORCE)
endif()

set(HOLOHUB_BUILD_OPERATORS "${holohub_build_operators}" CACHE STRING
    "Operators to build dynamically" FORCE)

if(HOLOHUB_DOWNLOAD_DATASETS)
  if(NOT DEFINED HOLOHUB_DATA_DIR OR HOLOHUB_DATA_DIR STREQUAL "")
    message(FATAL_ERROR "HOLOHUB_DATA_DIR must be set when building with datasets")
  endif()

  include(holoscan_download_data)

  if(NOT EXISTS "${HOLOHUB_DATA_DIR}/atracsys_visualizer/ir_base.gxf_entities")
    holoscan_download_data(atracsys_visualizer
      URL https://github.com/waylandio/holohub/releases/download/dataset-v1.0/atracsys_visualizer.zip
      DOWNLOAD_NAME atracsys_visualizer.zip
      DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
        ALL
    )
  endif()
endif()

if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()
