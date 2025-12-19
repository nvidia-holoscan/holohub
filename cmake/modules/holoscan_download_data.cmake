# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Creates a target with the <dataname>_data
function(holoscan_download_data dataname)

  # If we already have the target we return
  if(TARGET "${dataname}_data")
    return()
  endif()

  cmake_parse_arguments(DATA "GENERATE_GXF_ENTITIES;ALL;MODEL"
                             "URL;URL_MD5;DOWNLOAD_DIR;DOWNLOAD_NAME;GXF_ENTITIES_WIDTH;GXF_ENTITIES_HEIGHT;GXF_ENTITIES_CHANNELS;GXF_ENTITIES_FRAMERATE"
                             "" ${ARGN})

  if(NOT DATA_URL)
    message(FATAL_ERROR "No URL set for holoscan_download_data. Please set the URL.")
  endif()

  if(NOT DATA_DOWNLOAD_DIR)
    message(FATAL_ERROR "No DOWNLOAD_DIR set for holoscan_download_data. Please set the DOWNLOAD_DIR.")
  endif()

  if(NOT DATA_DOWNLOAD_NAME)
    set(DATA_DOWNLOAD_NAME ${dataname})
  endif()

  if(DATA_URL_MD5)
    list(APPEND extra_data_options --md5 ${DATA_URL_MD5})
  endif()

  if(DATA_GENERATE_GXF_ENTITIES)
     list(APPEND extra_data_options --generate_gxf_entities)
  endif()

  if(DATA_GXF_ENTITIES_WIDTH)
    list(APPEND extra_data_options --gxf_entities_width ${DATA_GXF_ENTITIES_WIDTH})
  endif()

  if(DATA_GXF_ENTITIES_HEIGHT)
    list(APPEND extra_data_options --gxf_entities_height ${DATA_GXF_ENTITIES_HEIGHT})
  endif()

  if(DATA_GXF_ENTITIES_CHANNELS)
    list(APPEND extra_data_options --gxf_entities_channels ${DATA_GXF_ENTITIES_CHANNELS})
  endif()

  if(DATA_GXF_ENTITIES_FRAMERATE)
    list(APPEND extra_data_options --gxf_entities_framerate ${DATA_GXF_ENTITIES_FRAMERATE})
  endif()

  if(DATA_MODEL)
    list(APPEND extra_data_options --model)
  endif()

  # Using a custom_command attached to a custom target allows to run only the custom command
  # if the stamp is not generated
  add_custom_command(OUTPUT "${DATA_DOWNLOAD_DIR}/${dataname}/${DATA_DOWNLOAD_NAME}.stamp"
     COMMAND ${CMAKE_SOURCE_DIR}/utilities/download_ngc_data
     --url ${DATA_URL}
     --dataset_name ${dataname}
     --download_dir ${DATA_DOWNLOAD_DIR}
     --download_name ${DATA_DOWNLOAD_NAME}
     ${extra_data_options}
     WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/utilities"
  )

  # If the target should be run all the time
  set(ALL)
  if(DATA_ALL)
    set(ALL "ALL")
  endif()

  add_custom_target("${dataname}_data" ${ALL} DEPENDS "${DATA_DOWNLOAD_DIR}/${dataname}/${DATA_DOWNLOAD_NAME}.stamp")

endfunction()

# Helper function to download models for NGC
function(holoscan_download_model modelname)
  holoscan_download_data(${modelname} MODEL ${ARGN})
endfunction()
