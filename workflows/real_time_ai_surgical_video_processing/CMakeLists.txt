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
project(real_time_ai_surgical_video_processing_workflow LANGUAGES NONE)

# Create a custom target that runs the Python script
add_custom_target(real_time_ai_surgical_video_processing
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/python/main.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Make sure Python interpreter is found
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
    GXF_ENTITIES_WIDTH 864
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )

  holoscan_download_data(detection_model
    URL nvidia/clara-holoscan/ssd_surgical_tool_detection_model:v0.1
    DOWNLOAD_NAME ssd_surgical_tool_detection_model_v0.1.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )

  holoscan_download_data(segmentation_model
    URL nvidia/clara-holoscan/monai_endoscopic_tool_segmentation_model:v0.1
    DOWNLOAD_NAME ll.1.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )

  holoscan_download_data(endoscopy_out_of_body_detection
    URL nvidia/clara-holoscan/endoscopy_out_of_body_detection:20230127
    DOWNLOAD_NAME endoscopy_out_of_body_detection_20230127.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 256
    GXF_ENTITIES_WIDTH 256
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )

  holoscan_download_data(orsi
    URL  nvidia/clara-holoscan/holoscan_orsi_academy_sample_data:20240206
    DOWNLOAD_NAME orsi.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 1920
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
endif()
