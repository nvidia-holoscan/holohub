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

cmake_minimum_required(VERSION 3.24)
project(grpc_hello_world LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/" "/workspace/holoscan-sdk/install")

include(protos/common.cmake)

set(CORE_PROTO_FILES
  ${CMAKE_CURRENT_LIST_DIR}/protos/holoscan.proto
)
grpc_generate_cpp(CORE_GRPC_SRCS CORE_GRPC_HDRS INCLUDE_DIRS ${CORE_PROTO_FILES})

add_library(grpc_hello_world_protos ${CORE_GRPC_HDRS} ${CORE_GRPC_SRCS})

target_link_libraries(grpc_hello_world_protos
  ${GRPC_GRPCPP}
  ${GRPCPP_REFLECTION}
  ${PROTOBUF_LIBPROTOBUF})

add_executable(grpc_hello_world main.cpp)
target_include_directories(grpc_hello_world PUBLIC ${INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS})


target_link_libraries(grpc_hello_world
    absl::flags
    absl::flags_parse
    ${GRPC_GRPCPP}
    ${GRPCPP_REFLECTION}
    ${PROTOBUF_LIBPROTOBUF}
    holoscan::core
    grpc_hello_world_protos)

# Copy the config to the binary directory
add_custom_target(grpc_hello_world_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/grpc_hello_world.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "grpc_hello_world.yaml"
  BYPRODUCTS "grpc_hello_world.yaml"
)
add_dependencies(grpc_hello_world grpc_hello_world_deps)