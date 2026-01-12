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

include(FetchContent)

set(NATS_BUILD_WITH_TLS OFF)
set(NATS_BUILD_STREAMING OFF)
set(NATS_BUILD_STATIC OFF)
set(NATS_BUILD_SHARED ON)
set(NATS_BUILD_EXAMPLES OFF)
FetchContent_Declare(
  nats-c
  GIT_REPOSITORY https://github.com/nats-io/nats.c.git
  GIT_TAG v3.10.1
)
FetchContent_MakeAvailable(nats-c)

# Create C++ NATS logger library
add_library(nats_logger SHARED
  nats_logger.cpp
  nats_logger.hpp
  create_tensor.cpp
)

target_include_directories(nats_logger PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nats_logger PRIVATE
  holoscan::core
  nats
  pipeline_visualization_flatbuffers::target
)

# Create example application using the NATS logger
add_executable(pipeline_visualization
  pipeline_visualization.cpp
)

target_link_libraries(pipeline_visualization PRIVATE
  nats_logger
  holoscan::core
)

# Copy config file
add_custom_target(pipeline_visualization_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_visualization.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "pipeline_visualization.yaml"
  BYPRODUCTS "pipeline_visualization.yaml"
)

add_dependencies(pipeline_visualization pipeline_visualization_yaml)
