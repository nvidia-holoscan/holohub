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

# Download and extract NVIDIA Video Codec SDK
set(NVC_SDK_URL "https://edge.urm.nvidia.com:443/artifactory/sw-holoscan-cli-generic/holohub/operators/holohub_nvc_13.0.19.zip")
set(NVC_SDK_ZIP "${CMAKE_BINARY_DIR}/holohub_nvc_13.0.19.zip")
set(NVC_SDK_DIR "${CMAKE_BINARY_DIR}/_deps/nvc_sdk")

# Download the SDK if it doesn't exist
if(NOT EXISTS ${NVC_SDK_ZIP})
    file(DOWNLOAD ${NVC_SDK_URL} ${NVC_SDK_ZIP} SHOW_PROGRESS EXPECTED_MD5 "a08c33c282bc92469ab8cf863aca9087")
endif()

# Extract the SDK if it hasn't been extracted
if(NOT EXISTS ${NVC_SDK_DIR})
    file(MAKE_DIRECTORY ${NVC_SDK_DIR})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${NVC_SDK_ZIP}
        WORKING_DIRECTORY ${NVC_SDK_DIR}
    )
endif()

# Set the SDK path for the operators to use
set(NVC_SDK_PATH ${NVC_SDK_DIR} CACHE PATH "Path to NVIDIA Video Codec SDK")


add_holohub_operator(nv_video_encoder)
add_holohub_operator(nv_video_decoder)
add_holohub_operator(nv_video_reader)