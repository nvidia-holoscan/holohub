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
project(grpc_endoscopy_tool_tracking CXX)

find_package(holoscan 2.6 REQUIRED CONFIG
  PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(grpc_endoscopy_tool_tracking_cloud
  cloud/app_cloud_main.cpp
  cloud/grpc_service.hpp
  cloud/app_cloud_pipeline.hpp
)

add_executable(grpc_endoscopy_tool_tracking_edge
  edge/app_edge_main.cpp
  edge/app_edge_single_fragment.hpp
  edge/app_edge_multi_fragment.hpp
  edge/video_input_fragment.hpp
  edge/viz_fragment.hpp
)

target_link_libraries(grpc_endoscopy_tool_tracking_cloud
  PRIVATE
  holoscan::core
  holoscan::ops::format_converter
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
  grpc_operators
)

target_link_libraries(grpc_endoscopy_tool_tracking_edge
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
  grpc_operators
)

# Copy the config to the binary directory
add_custom_target(grpc_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml"
)
add_dependencies(grpc_endoscopy_tool_tracking_edge grpc_endoscopy_tool_tracking_yaml)

# Copy the launch script
add_custom_target(grpc_endoscopy_tool_tracking_launch_sh
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/launch.sh" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "launch.sh"
  BYPRODUCTS "launch.sh"
)
add_dependencies(grpc_endoscopy_tool_tracking_edge grpc_endoscopy_tool_tracking_launch_sh)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20230222/zip
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    URL_MD5 d54f84a562d29ed560a87d2607eba973
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
  add_dependencies(grpc_endoscopy_tool_tracking_edge endoscopy_data)
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS grpc_endoscopy_tool_tracking_edge
  DESTINATION bin/grpc_endoscopy_tool_tracking/cpp
)

install(
  TARGETS grpc_endoscopy_tool_tracking_cloud
  DESTINATION bin/grpc_endoscopy_tool_tracking/cpp
)

install(
  FILES endoscopy_tool_tracking.yaml
  DESTINATION bin/grpc_endoscopy_tool_tracking/cpp
)
