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
project(orsi_sample_apps LANGUAGES NONE)

# Check if Deltacast is found
set(ORSI_VIDEO_MASTER_SDK_FOUND OFF)

if(IS_DIRECTORY "/usr/local/deltacast/Include")
    if(EXISTS "/usr/lib/libvideomasterhd.so")
        set(ORSI_VIDEO_MASTER_SDK_FOUND ON)
    endif()
endif()

set(HOLOSCAN_OPERATORS
        orsi_format_converter
        orsi_segmentation_postprocessor
        orsi_segmentation_preprocessor
        orsi_visualizer
)

if(ORSI_VIDEO_MASTER_SDK_FOUND)
    list(APPEND HOLOSCAN_OPERATORS deltacast_videomaster)
endif()

# Download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download ORSI sample data from NGC
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(orsi
    URL  nvidia/clara-holoscan/holoscan_orsi_academy_sample_data:20240206
    DOWNLOAD_NAME orsi.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 1920
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
  )
endif()

# Only configure the lib if the applications are build
if(APP_orsi_multi_ai_ar OR APP_orsi_in_out_body OR APP_orsi_segmentation_ar)
  add_subdirectory(lib)
endif()

add_holohub_application(orsi_multi_ai_ar DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS}
                        OPTIONAL aja_source)

add_holohub_application(orsi_in_out_body DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS}
                        OPTIONAL aja_source)

add_holohub_application(orsi_segmentation_ar DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS}
                        OPTIONAL aja_source)
