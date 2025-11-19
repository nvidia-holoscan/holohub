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

# messages/schemas

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(isaac_generate_fbs_headers)
isaac_generate_fbs_headers(
  ISAAC_MESSAGES_GENERATED_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(isaac_messages_header_generation
  DEPENDS ${ISAAC_MESSAGES_GENERATED_HEADERS}
)

add_library(isaac_messages_schemas INTERFACE)
add_library(isaac::messages::schemas ALIAS isaac_messages_schemas)

target_include_directories(isaac_messages_schemas
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # Includes of source headers are relative to the source root dir.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(isaac_messages_schemas
  INTERFACE
    flatbuffers::flatbuffers
    holoscan::core
)
add_dependencies(isaac_messages_schemas isaac_messages_header_generation)

# messages/reflection

# isaac::messages::reflection - registry and reflection helpers.

add_library(isaac_messages_reflection
  message_registry.cc
)
add_library(isaac::messages::reflection ALIAS isaac_messages_reflection)

target_include_directories(isaac_messages_reflection
  PUBLIC ${CMAKE_SOURCE_DIR}
)

target_link_libraries(isaac_messages_reflection
  PUBLIC
    flatbuffers::flatbuffers
    isaac::messages::schemas
)

add_library(isaac_messages_reflection_tensor_materialization INTERFACE)
add_library(isaac::messages::reflection::tensor_materialization ALIAS isaac_messages_reflection_tensor_materialization)

target_include_directories(isaac_messages_reflection_tensor_materialization
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
