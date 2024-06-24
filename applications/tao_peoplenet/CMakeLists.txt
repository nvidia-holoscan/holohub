# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(tao_peoplenet NONE)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the model
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tao_peoplenet/resnet34_peoplenet_int8.onnx"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/tao_peoplenet"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/tao_peoplenet/resnet34_peoplenet_int8.onnx"
              -L "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/tao_peoplenet/resnet34_peoplenet_int8.onnx"
    VERBATIM
)

# Download the testing video
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.mp4"
    COMMAND mkdir -p "${HOLOHUB_DATA_DIR}/tao_peoplenet"
    COMMAND curl -S -o "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.mp4"
              -L "https://www.pexels.com/download/video/5385885/?fps=25.0&h=1920&w=1080"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.mp4"
    VERBATIM
)

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.gxf_index"
                          "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/tao_peoplenet" --basename twirl --width 1080 --height 1920 --framerate 25
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.gxf_index"
               "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.gxf_entities"
    DEPENDS "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.mp4"
)


add_custom_target(tao_peoplenet_data ALL
    DEPENDS
    "${HOLOHUB_DATA_DIR}/tao_peoplenet/resnet34_peoplenet_int8.onnx"
    "${HOLOHUB_DATA_DIR}/tao_peoplenet/twirl.gxf_index"
)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/tao_peoplenet.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tao_peoplenet_testing.yaml ${CONFIG_FILE})

  add_custom_target(tao_peoplenet_test ALL
    DEPENDS "tao_peoplenet.py"
  )

  # Add test
  add_test(NAME tao_peoplenet_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tao_peoplenet.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/tao_peoplenet_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/tao_peoplenet"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST tao_peoplenet_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(tao_peoplenet_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
