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

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  set(DOWNLOAD_DATA_DIR ${HOLOHUB_DATA_DIR}/imaging_ai_segmentator)

  file(MAKE_DIRECTORY ${DOWNLOAD_DATA_DIR}/models)
  file(MAKE_DIRECTORY ${DOWNLOAD_DATA_DIR}/dicom)
  add_custom_command(
    OUTPUT imaging_ai_segmentator_data
    COMMAND gdown -q https://drive.google.com/uc?id=1PHpFWboimEXmMSe2vBra6T8SaCMC2SHT -O ${DOWNLOAD_DATA_DIR}/models/model.pt
    COMMENT "Downloading imaging_ai_segmentator model using gdown..."
  )
  add_custom_target(run ALL
      DEPENDS imaging_ai_segmentator_data)
endif()