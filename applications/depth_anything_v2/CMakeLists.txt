# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(depth_anything_v2 NONE)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Copy depth anything v2 model if it's not in the data directory
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/depth_anything_v2_vits.onnx"
    COMMAND mkdir -p ${HOLOHUB_DATA_DIR}/depth_anything_v2/
    COMMAND cp "/models/depth_anything_v2_vits.onnx" "${HOLOHUB_DATA_DIR}/depth_anything_v2/"
    )

# Download the testing video and trim
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/depth_anything_v2"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4"
              -L "https://www.pexels.com/download/video/5823544/?fps=50.0&h=720&w=1280"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4" -t 6.4
              -c:v libx264 "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    COMMAND rm -rf "${HOLOHUB_DATA_DIR}/depth_anything_v2/video.mp4"
    VERBATIM
    )


# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_index"
                          "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/depth_anything_v2" --basename running_on_pathway --width 1280 --height 720 --framerate 50
    DEPENDS "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.mp4"
    )

add_custom_target(depth_anything_v2_data ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/depth_anything_v2/depth_anything_v2_vits.onnx"
                  "${HOLOHUB_DATA_DIR}/depth_anything_v2/running_on_pathway.gxf_index")

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/depth_anything_v2.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/depth_anything_v2_testing.yaml ${CONFIG_FILE})

  add_custom_target(depth_anything_v2_test ALL
    DEPENDS "depth_anything_v2.py"
  )

  # Add test
  add_test(NAME depth_anything_v2_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/depth_anything_v2.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/depth_anything_v2_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/depth_anything_v2"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST depth_anything_v2_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(depth_anything_v2_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
