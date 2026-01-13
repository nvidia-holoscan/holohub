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

cmake_minimum_required(VERSION 3.20)

project(ultrasound_postprocessing)

if(HOLOHUB_DOWNLOAD_DATASETS)
    # Custom target for downloading raw .uff file
    set(UFF_URL "http://www.ustb.no/datasets/PICMUS_simulation_contrast_speckle.uff")
    set(UFF_SHA256 "be3d1efb0104dc499826e981187bea5520d17373aad69370961ae6318dd877f9")

    set(UFF_DOWNLOAD_DIR "${HOLOHUB_DATA_DIR}/ultrasound_postprocessing")
    set(UFF_OUTPUT_FILE "${UFF_DOWNLOAD_DIR}/demo.uff")
    set(UFF_STAMP_FILE "${UFF_DOWNLOAD_DIR}/demo.uff.stamp")

    # Find download command
    find_program(WGET_COMMAND wget)
    find_program(CURL_COMMAND curl)
    find_program(SHA256_COMMAND sha256sum)

    if(WGET_COMMAND)
        set(DOWNLOAD_CMD ${WGET_COMMAND} -q --show-progress -O ${UFF_OUTPUT_FILE} ${UFF_URL})
    elseif(CURL_COMMAND)
        set(DOWNLOAD_CMD ${CURL_COMMAND} -S -L -o ${UFF_OUTPUT_FILE} ${UFF_URL})
    else()
        message(FATAL_ERROR "wget or curl is required to download datasets")
    endif()

    add_custom_command(
        OUTPUT ${UFF_STAMP_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${UFF_DOWNLOAD_DIR}
        COMMAND ${DOWNLOAD_CMD}
        COMMAND ${CMAKE_COMMAND} -E touch ${UFF_STAMP_FILE}
        COMMAND ${SHA256_COMMAND} ${UFF_OUTPUT_FILE} | grep -q ${UFF_SHA256} || (echo "UFF download checksum failed!" && exit 1)
        COMMENT "Downloading demo.uff from ustb.no"
    )

    add_custom_target(ultrasound_postprocessing_data ALL
        DEPENDS ${UFF_STAMP_FILE}
    )
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
