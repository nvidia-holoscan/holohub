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
project(multiai_endoscopy_apps LANGUAGES NONE)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20230222/zip
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    URL_MD5 d54f84a562d29ed560a87d2607eba973
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
  )

  holoscan_download_data(ssd_model
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/ssd_surgical_tool_detection_model/versions/v0.1/zip
    DOWNLOAD_NAME ssd_surgical_tool_detection_model_v0.1.zip
    URL_MD5 daf3663b4bc56de1cb324b0e58aed450
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )

  holoscan_download_data(monai_tool_seg_model
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/monai_endoscopic_tool_segmentation_model/versions/v0.1/zip
    DOWNLOAD_NAME monai_endoscopic_tool_segmentation_model_v0.1.zip
    URL_MD5 222d924c5b2b0f196a4ef605c9977764
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
endif()

add_subdirectory(cpp)
add_subdirectory(python)
