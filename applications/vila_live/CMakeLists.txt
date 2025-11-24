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
project(vila_live NONE)

message(STATUS "Starting vila_live setup...")

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the testing video and trim
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/vila_live/meeting.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/vila_live"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/vila_live/video.mp4"
              -L "https://www.pexels.com/download/video/3252919/?fps=25.0&h=720&w=1280"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/vila_live/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/vila_live/meeting.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/vila_live/video.mp4"
    VERBATIM
    )

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/vila_live/meeting.gxf_index"
                          "${HOLOHUB_DATA_DIR}/vila_live/meeting.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/vila_live/meeting.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/vila_live" --basename meeting --width 1280 --height 720 --framerate 25
    DEPENDS "${HOLOHUB_DATA_DIR}/vila_live/meeting.mp4"
    )

add_custom_target(vila_live ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/vila_live/meeting.gxf_index")

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vila_live.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  # add model path and quant path to the yaml file, see also Dockerfile
  string(APPEND CONFIG_FILE "\nmodel_path: /workspace/volumes/models/Llama-3-VILA1.5-8b-Fix-AWQ/")
  string(APPEND CONFIG_FILE "\nquant_path: /workspace/volumes/models/Llama-3-VILA1.5-8b-Fix-AWQ/llm/llama-3-vila1.5-8b-fix-w4-g128-awq-v2.pt")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vila_live_testing.yaml ${CONFIG_FILE})

  add_custom_target(vila_live_test ALL
    DEPENDS "vila_live"
  )
  add_subdirectory(tests)

endif()
