# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Find holoscan to ensure CMAKE_CUDA_COMPILER is consumed (silences CMake warning
# about unused variable for Python-only apps with project(NONE))
find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)

  holoscan_download_model(polyp_detection
    URL nvidia/clara-holoscan/polyp_detection_rt_detr_model:20250304
    DOWNLOAD_NAME polyp_detection_rt_detr_model_20250304.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )

  holoscan_download_data(colonoscopy_segmentation
    URL nvidia/clara-holoscan/holoscan_colonoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_colonoscopy_sample_data_20230222.zip
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
                   --data "${HOLOHUB_DATA_DIR}/polyp_detection/polyp_detection_rt_detr_model_v20250304"
                   --video_dir "${HOLOHUB_DATA_DIR}/polyp_detection"
                   --source replayer
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST polyp_detection_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib")

  set_tests_properties(polyp_detection_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
