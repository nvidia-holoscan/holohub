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

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the associated dataset if needed
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(colonoscopy_segmentation
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_colonoscopy_sample_data/versions/20230222/zip
    DOWNLOAD_NAME holoscan_colonoscopy_sample_data_20230222.zip
    URL_MD5 46e42fb08602fa1acdd1c02f10bde5a5
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 576
    GXF_ENTITIES_WIDTH 720
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
    )
endif()

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 0.6 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy_segmentation.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/colonoscopy_segmentation_testing.yaml ${CONFIG_FILE})

  # Add test
  add_test(NAME colonoscopy_segmentation_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy_segmentation.py 
                   --config ${CMAKE_CURRENT_BINARY_DIR}/colonoscopy_segmentation_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/colonoscopy_segmentation"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST colonoscopy_segmentation_python_test PROPERTY ENVIRONMENT 
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")  
  
  set_tests_properties(colonoscopy_segmentation_python_test 
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
