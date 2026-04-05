# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  set(OP_atracsys_camera ON CACHE BOOL "Build the atracsys_camera operator" FORCE)
  set(HOLOHUB_BUILD_OPERATORS "atracsys_camera" CACHE STRING
      "Optional operators requested for the active HoloHub app build" FORCE)
else()
  set(OP_atracsys_camera OFF CACHE BOOL "Build the atracsys_camera operator" FORCE)
  set(HOLOHUB_BUILD_OPERATORS "" CACHE STRING
      "Optional operators requested for the active HoloHub app build" FORCE)
endif()



if(HOLOHUB_BUILD_CPP)
  add_subdirectory(cpp)
endif()
