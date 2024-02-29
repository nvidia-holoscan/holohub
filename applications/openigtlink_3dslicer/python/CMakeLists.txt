# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Download the ultrasound sample data
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

  add_dependencies(openigtlink_3dslicer colonoscopy_segmentation_data)
endif()