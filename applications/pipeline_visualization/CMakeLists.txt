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

include(FetchContent)

find_package(holoscan 3.8.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_BUILD_FLATHASH OFF)
set(FLATBUFFERS_BUILD_FLATC ON)
set(FLATBUFFERS_BUILD_FLATLIB ON)
set(FLATBUFFERS_STATIC_FLATC ON)
set(FLATBUFFERS_STRICT_MODE ON)
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v25.9.23
)
FetchContent_MakeAvailable(flatbuffers)
list(APPEND CMAKE_MODULE_PATH "${flatbuffers_SOURCE_DIR}/CMake")

include(BuildFlatBuffers)

set(FB_SCHEMA_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/schemas/message.fbs"
    "${CMAKE_CURRENT_SOURCE_DIR}/schemas/tensor.fbs"
    )

# Build for cpp and python
set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS
    --cpp
    --python
    --bfbs-gen-embed
    --cpp-ptr-type std::shared_ptr
    --gen-object-api
    --reflect-names
    --reflect-types
    --schema
    )

build_flatbuffers(
    "${FB_SCHEMA_FILES}" # flatbuffers_schemas
    "" # flatbuffers_options
    "pipeline_visualization_flatbuffers_schemas" # flatbuffers_target_name
    "" # additional_dependencies
    "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers" # generated_includes_dir
    "" # binary_schemas_dir
    "" # copy_text_schemas_dir
    )

add_library(pipeline_visualization_flatbuffers_target INTERFACE)
add_library(pipeline_visualization_flatbuffers::target ALIAS pipeline_visualization_flatbuffers_target)
target_include_directories(pipeline_visualization_flatbuffers_target
    INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${flatbuffers_SOURCE_DIR}/include
    )
add_dependencies(pipeline_visualization_flatbuffers_target
    pipeline_visualization_flatbuffers_schemas
    )

add_subdirectory(cpp)
if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
