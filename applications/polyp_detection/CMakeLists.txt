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

option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# download ngc data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(polyp_detection_sample_app_data
    URL nvstaging/holoscan/polyp_detection_sample_app_data:1.0.0
    DOWNLOAD_NAME polyp_detection_sample_app_data_1.0.0.zip
    DOWNLOAD_DIR "${HOLOHUB_DATA_DIR}/polyp_detection"
    ALL
    )
endif()

# install extra dependencies
add_custom_target(install_python_dependencies ALL
    COMMAND python3 -m pip install torch torchvision
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Convert the video file
add_custom_command(OUTPUT "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/tensor.gxf_index"
                          "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/tensor.gxf_entities"
    COMMAND ffmpeg -i "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/004-015_first_900_original.mp4" -pix_fmt rgb24 -f rawvideo pipe:1 |
            python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
            --directory "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data" --basename tensor --width 1164 --height 1034 --channels 3 --framerate 30
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/tensor.gxf_index"
               "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/tensor.gxf_entities"
    DEPENDS "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/004-015_first_900_original.mp4"
    )

add_custom_target(polyp_detection ALL
    DEPENDS "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_sample_app_data/tensor.gxf_index")

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
