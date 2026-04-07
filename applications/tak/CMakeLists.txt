# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(tak NONE)

find_package(holoscan 3.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the YOLOv8s model (.pt for ultralytics)
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tak/yolov8s.pt"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/tak"
    COMMAND wget -q -O "${HOLOHUB_DATA_DIR}/tak/yolov8s.pt"
              "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    VERBATIM
    )

# Download the sample video and trim to 6.4 seconds
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tak/traffic.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/tak"
    COMMAND curl -S -L -o "${HOLOHUB_DATA_DIR}/tak/video.mp4"
              "https://www.pexels.com/download/video/19326803/?fps=29.97&h=720&w=1280"
    COMMAND ffmpeg -y -i "${HOLOHUB_DATA_DIR}/tak/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/tak/traffic.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/tak/video.mp4"
    VERBATIM
    )

# Convert the video file to GXF entities for the replayer
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tak/traffic.gxf_index"
                          "${HOLOHUB_DATA_DIR}/tak/traffic.gxf_entities"
    COMMAND sh -c "ffmpeg -i \"${HOLOHUB_DATA_DIR}/tak/traffic.mp4\" -pix_fmt rgb24 -f rawvideo pipe:1 | python3 \"${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py\" --directory \"${HOLOHUB_DATA_DIR}/tak\" --basename traffic --width 1280 --height 720 --framerate 30"
    DEPENDS "${HOLOHUB_DATA_DIR}/tak/traffic.mp4"
    VERBATIM
    )

add_custom_target(tak_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/tak/yolov8s.pt"
                  "${HOLOHUB_DATA_DIR}/tak/traffic.gxf_index")

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing (limit frame count)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tak.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tak_testing.yaml ${CONFIG_FILE})

  add_custom_target(tak_test ALL
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tak.py"
            tak_data
  )

  add_test(NAME tak_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tak.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/tak_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/tak"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST tak_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(tak_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
