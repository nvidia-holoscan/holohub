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

cmake_minimum_required(VERSION 3.20)
project(stereo_vision_app)

# Download the stereo vision sample video
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(stereo_vision
    URL nvidia/clara-holoscan/holoscan_stereo_video:20241216
    DOWNLOAD_NAME holoscan_stereo_vision_20241216.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 9120
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
endif()

add_subdirectory(cpp)
