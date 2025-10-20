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

cmake_minimum_required(VERSION 3.20)
project(nvc_decode_cpp)

find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(nvc_decode_cpp
  nvc_decode.cpp
)

target_link_libraries(nvc_decode_cpp
  holoscan::core
  holoscan::ops::holoviz
  nv_video_decoder
  nv_video_reader
)

# Copy config file to executable output directory
add_custom_command(TARGET nvc_decode_cpp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/nvc_decode.yaml"
                                   "$<TARGET_FILE_DIR:nvc_decode_cpp>/nvc_decode.yaml"
)

# Set install rules
install(TARGETS nvc_decode_cpp
        DESTINATION "bin"
        COMPONENT "holoscan-examples")

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/nvc_decode.yaml"
        DESTINATION "share/holohub/applications/nvidia_video_codec/nvc_decode/"
        COMPONENT "holoscan-examples")
