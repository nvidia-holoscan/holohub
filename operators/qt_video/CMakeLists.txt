# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(qt_video)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick)

if(COMMAND qt_standard_project_setup)
  qt_standard_project_setup()
else()
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
endif()

qt_add_library(qt_video SHARED
  opengl_renderer.cpp
  opengl_renderer.hpp
  qt_holoscan_video.cpp
  qt_holoscan_video.hpp
  qt_video_op.cpp
  qt_video_op.hpp
  shared_data.hpp
  )

add_library(holoscan::ops::qt_video ALIAS qt_video)

target_link_libraries(qt_video
  PRIVATE
    holoscan::core
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
)

qt_add_qml_module(qt_video
    URI QtHoloscanVideo
    VERSION 1.0
  )


target_include_directories(qt_video INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(qt_video PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(qt_video PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})
