# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(orsi_app_lib CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(orsi_app_lib EXCLUDE_FROM_ALL
    orsi_app.hpp
    orsi_app.cpp
)

target_link_libraries(orsi_app_lib
   PUBLIC
   holoscan::core
   holoscan::ops::video_stream_replayer
   holoscan::orsi::format_converter
)

target_link_libraries(orsi_app_lib PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::aja>)

set(VIDEOMASTER_OPERATOR "")
if(ORSI_VIDEO_MASTER_SDK_FOUND)
    target_compile_definitions(orsi_app_lib PRIVATE -DUSE_VIDEOMASTER)
    target_link_libraries(orsi_app_lib PRIVATE holoscan::videomaster)
endif()

target_include_directories(orsi_app_lib
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)