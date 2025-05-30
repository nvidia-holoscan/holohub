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

cmake_minimum_required(VERSION 3.20)
project(imaging_ai_segmentator CXX)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Install application and dependencies into the install/ directory for packaging
install(
  FILES __main__.py
        app.py
        env_settings.sh
        monai_totalseg_operator.py
        README.md
        requirements.txt
        app.yaml
  DESTINATION imaging_ai_segmentator
  COMPONENT "holohub-apps"
)

install(
  DIRECTORY ${CMAKE_HOME_DIRECTORY}/operators/medical_imaging
  DESTINATION imaging_ai_segmentator/operators)

# Download the associated datasets and models if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  set(DOWNLOAD_DATA_DIR ${HOLOHUB_DATA_DIR}/imaging_ai_segmentator)

  # Download the MONAI model
  file(MAKE_DIRECTORY ${DOWNLOAD_DATA_DIR}/models)
  add_custom_command(
    OUTPUT imaging_ai_segmentator_model
    COMMAND  wget https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_wholebody_ct_segmentation.pt -O ${DOWNLOAD_DATA_DIR}/models/model.pt
    COMMENT "Downloading imaging_ai_segmentator model from MONAI Model Zoo (hosted by NVIDIA)..."
  )

  # Download a sample CT DICOM Image
  holoscan_download_data(imaging_ai_segmentator
      URL https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/CT_DICOM_SINGLE.zip
      DOWNLOAD_DIR ${DOWNLOAD_DATA_DIR}/dicom
      ALL
  )

  add_custom_target(run ALL
      DEPENDS imaging_ai_segmentator_model imaging_ai_segmentator_data)
endif()
