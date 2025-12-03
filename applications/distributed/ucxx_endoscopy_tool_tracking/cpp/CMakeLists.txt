# SPDX-FileCopyrightText: Copyright (c) 2025 Holoscan Team
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

project(ucxx_endoscopy_tool_tracking)

find_package(holoscan 3.9.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_executable(ucxx_endoscopy_tool_tracking
    main.cpp
    server.cpp
    client.cpp
)

target_include_directories(ucxx_endoscopy_tool_tracking
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ucxx_endoscopy_tool_tracking
    PRIVATE
    holoscan::core
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::ucxx
    lstm_tensor_rt_inference
    tool_tracking_postprocessor
)

# Copy configuration file
add_custom_target(ucxx_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ucxx_endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ucxx_endoscopy_tool_tracking.yaml"
)

add_dependencies(ucxx_endoscopy_tool_tracking ucxx_endoscopy_tool_tracking_yaml)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
  add_dependencies(ucxx_endoscopy_tool_tracking endoscopy_data)
endif()


if(BUILD_TESTING)
  add_test(ucxx_endoscopy_tool_tracking-test
    COMMAND echo "Add test coverage for ucxx_endoscopy_tool_tracking here"
  )
endif()

