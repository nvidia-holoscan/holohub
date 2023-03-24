# SPDX-FileCopyrightText: Copyright (c) 2022 DELTACAST.TV. All rights reserved.
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
project(videomaster)

find_package(holoscan 0.4 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# CPP Applications
add_executable(videomaster_replayer
  main.cpp
)

target_link_libraries(videomaster_replayer
  PRIVATE
  holoscan::holoscan
  holoscan::videomaster
)

# Copy config file
add_custom_target(videomaster_replayer_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "app_config.yaml"
  BYPRODUCTS "app_config.yaml"
)
add_dependencies(videomaster_replayer videomaster_replayer_yaml)


# GXF Applications
list(APPEND videomaster_COMMON_EXTENSIONS
  GXF::std
  GXF::cuda
  GXF::multimedia
  GXF::serialization
  format_converter
  stream_playback
)

create_gxe_application(
  NAME videomaster_replayer
  YAML app_config.yaml
  EXTENSIONS
    ${videomaster_COMMON_EXTENSIONS}
    holoscan::exts::gxf_videomaster
    custom_lstm_inference
    visualizer_tool_tracking
)