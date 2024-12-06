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
project(grpc_operators)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

include(grpc_generate_cpp)

set(CORE_PROTO_FILES
  ${CMAKE_CURRENT_LIST_DIR}/protos/holoscan.proto
)
grpc_generate_cpp(CORE_GRPC_SRCS CORE_GRPC_HDRS INCLUDE_DIRS ${CORE_PROTO_FILES})

add_library(grpc_operators SHARED
  ${CORE_GRPC_SRCS}
  server/application_factory.cpp
  server/entity_server.cpp
  server/grpc_server_request.cpp
  server/grpc_server_response.cpp
  server/grpc_application.cpp
  common/asynchronous_condition_queue.hpp
  common/conditional_variable_queue.hpp
  common/tensor_proto.cpp
  client/entity_client.cpp
  client/entity_client_service.cpp
  client/grpc_client_request.cpp
  client/grpc_client_response.cpp
)

target_link_libraries(grpc_operators
  PUBLIC
  ${GRPC_GRPCPP}
  ${GRPCPP_REFLECTION}
  ${PROTOBUF_LIBPROTOBUF}
  PRIVATE
  holoscan::core
  GXF::std
  )

target_include_directories(grpc_operators
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS})

# Installation
install(TARGETS grpc_operators)

