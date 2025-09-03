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
project(nvc_encode_decode_cpp)

find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(nvc_encode_decode_cpp
  nvc_encode_decode.cpp
)

target_link_libraries(nvc_encode_decode_cpp
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  nv_video_encoder
  nv_video_decoder
)

# Copy config file to build tree
add_custom_target(nvc_encode_decode_cpp_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/nvc_encode_decode.yaml"
                                   "${CMAKE_CURRENT_BINARY_DIR}/nvc_encode_decode.yaml"
  DEPENDS "nvc_encode_decode.yaml"
  BYPRODUCTS "nvc_encode_decode.yaml"
)

add_dependencies(nvc_encode_decode_cpp nvc_encode_decode_cpp_yaml)

# Set install rules
install(TARGETS nvc_encode_decode_cpp
        DESTINATION "bin"
        COMPONENT "holoscan-examples")

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/nvc_encode_decode.yaml"
        DESTINATION "share/holohub/applications/nvidia_video_codec/nvc_encode_decode/"
        COMPONENT "holoscan-examples")
