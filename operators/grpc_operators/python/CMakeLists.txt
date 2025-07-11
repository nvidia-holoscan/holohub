# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(grpc_operators)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(CORE_PROTO_FILES
  ${CMAKE_CURRENT_LIST_DIR}/../protos/holoscan.proto
)

# Generate Python code from the proto files
include(grpc_generate_python)
grpc_generate_python(PYTHON_SRC ${CORE_PROTO_FILES})
add_custom_target(grpc_python ALL DEPENDS ${PYTHON_SRC})

set(build_generated_dir "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/grpc_operators/")
file(MAKE_DIRECTORY ${build_generated_dir})

# Copy to the Python directory in the build
foreach(python_file ${PYTHON_SRC})
    add_custom_command(
        TARGET grpc_python
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${python_file} ${build_generated_dir}
        COMMENT "Copying generated proto file ${python_file}..."
)
endforeach()

# Installation
install(TARGETS grpc_operators)

install(
  DIRECTORY server client common
  DESTINATION lib/operators/grpc_operators/python/
)