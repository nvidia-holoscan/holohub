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

project(polyp_detection NONE)

# download model
add_custom_target(download_polyp_detection_rt_detr_model ALL
    COMMAND ngc registry model download-version "nvstaging/holoscan/polyp_detection_rt_detr_model:20250304" --dest "${HOLOHUB_DATA_DIR}/polyp_detection"
    COMMENT "Downloading polyp detection model"
    )

# install extra dependencies
add_custom_target(install_python_dependencies ALL
    COMMAND python3 -m pip install -r requirements.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing python dependencies"
    DEPENDS download_polyp_detection_rt_detr_model
    )

# Download and convert the video file
add_custom_target(prepare_app_example_data ALL
    COMMAND python3 prepare_app_example_data.py --download_dir "${HOLOHUB_DATA_DIR}/polyp_detection" --file_name "001-014_frames.tar.gz" --video_name "001-014_sample.mp4" --video_frames 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Downloading and converting video file"
    DEPENDS install_python_dependencies
    )

# Convert the video file to gxf_entities
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/polyp_detection/tensor.gxf_index"
                          "${HOLOHUB_DATA_DIR}/polyp_detection/tensor.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/polyp_detection/001-014_sample.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/polyp_detection" --basename tensor --width 1352 --height 1080 --channels 3 --framerate 30
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/polyp_detection/tensor.gxf_index"
               "${HOLOHUB_DATA_DIR}/polyp_detection/tensor.gxf_entities"
    COMMENT "Converting video file to gxf_entities"
    DEPENDS prepare_app_example_data
    )

add_custom_target(polyp_detection ALL
    COMMENT "Building polyp detection application"
    DEPENDS "${HOLOHUB_DATA_DIR}/polyp_detection/tensor.gxf_index")

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/polyp_detection.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/polyp_detection_testing.yaml ${CONFIG_FILE})

  add_custom_target(polyp_detection_test ALL
    DEPENDS "polyp_detection.py"
  )

  # Add test
  add_test(NAME polyp_detection_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/polyp_detection.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/polyp_detection_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data"
                   --video_dir "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST polyp_detection_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(polyp_detection_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
