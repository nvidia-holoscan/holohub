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
    list(APPEND HOLOSCAN_OPERATORS  deltacast_videomaster)
endif()

add_subdirectory(lib)

add_holohub_application(orsi_multi_ai_ar DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS})

add_holohub_application(orsi_in_out_body DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS})

add_holohub_application(orsi_segmentation_ar DEPENDS
                        OPERATORS ${HOLOSCAN_OPERATORS})
