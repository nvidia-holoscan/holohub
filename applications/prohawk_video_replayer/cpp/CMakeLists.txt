# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(prohawk_video_replayer CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(prohawk_video_replayer
prohawk_video_replayer.cpp
)

target_link_libraries(prohawk_video_replayer
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
  holoscan::prohawk_video_processing
)

# Copy config file
add_custom_target(prohawk_video_replayer_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/prohawk_video_replayer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/prohawk_video_replayer.yaml"
)
add_dependencies(prohawk_video_replayer prohawk_video_replayer_yaml)

# Copy PAFSFilter file
add_custom_target(prohawk_video_replayer_PAFSFilter
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PAFSFilter.bin" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/PAFSFilter.bin"
)
add_dependencies(prohawk_video_replayer prohawk_video_replayer_PAFSFilter)

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
  )
  add_dependencies(prohawk_video_replayer endoscopy_data)
endif()
