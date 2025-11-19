# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FlatBuffers helper utilities for CMake.

# isaac_generate_fbs_headers(<out_var> <input_dir> <output_dir>).
# - <out_var>: Variable name to receive the list of generated headers.
# - <input_dir>: Directory containing .fbs files.
# - <output_dir>: Output directory for generated headers.
function(isaac_generate_fbs_headers OUT_VAR INPUT_DIR OUTPUT_DIR)
  # Get flatc
  get_target_property(FLATC_EXECUTABLE flatbuffers::flatc LOCATION)

  # Identify input files
  file(GLOB FBS_FILES RELATIVE ${INPUT_DIR} ${INPUT_DIR}/*.fbs)
  if(FBS_FILES STREQUAL "")
    set(${OUT_VAR} "" PARENT_SCOPE)
    message(WARNING "No .fbs files found in ${INPUT_DIR}")
    return()
  endif()

  # Defined output directory
  if(NOT OUTPUT_DIR)
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  if(NOT EXISTS ${OUTPUT_DIR})
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
  endif()

  set(GENERATED_HEADER_LIST "")
  foreach(SCHEMA_FILE IN LISTS FBS_FILES)
    get_filename_component(SCHEMA_NAME ${SCHEMA_FILE} NAME_WE)
    set(OUT_HEADER ${OUTPUT_DIR}/${SCHEMA_NAME}_generated.h)
    set(OUT_BFBS_HEADER ${OUTPUT_DIR}/${SCHEMA_NAME}_bfbs_generated.h)
    add_custom_command(
      OUTPUT ${OUT_HEADER} ${OUT_BFBS_HEADER}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
      COMMAND ${FLATC_EXECUTABLE}
              --cpp
              --cpp-ptr-type std::shared_ptr
              --gen-object-api
              --schema
              --bfbs-gen-embed
              --reflect-names
              --reflect-types
              -I ${INPUT_DIR}
              -o ${OUTPUT_DIR}
              ${INPUT_DIR}/${SCHEMA_FILE}
      DEPENDS ${INPUT_DIR}/${SCHEMA_FILE}
      COMMENT "Generating FlatBuffers for ${INPUT_DIR}/${SCHEMA_FILE}."
      VERBATIM
    )
    list(APPEND GENERATED_HEADER_LIST ${OUT_HEADER} ${OUT_BFBS_HEADER})
  endforeach()

  set(${OUT_VAR} ${GENERATED_HEADER_LIST} PARENT_SCOPE)
endfunction()
