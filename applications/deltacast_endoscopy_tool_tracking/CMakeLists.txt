# Copyright (c) 2022-2023, DELTACAST.TV.
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
cmake_minimum_required(VERSION 3.20)
project(deltacast_endoscopy_tool_tracking)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# CPP Applications
add_executable(deltacast_endoscopy_tool_tracking
  main.cpp
)

target_link_libraries(deltacast_endoscopy_tool_tracking
  PRIVATE
  holoscan::core
  holoscan::ops::aja
  holoscan::ops::video_stream_replayer
  holoscan::ops::video_stream_recorder
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
  holoscan::videomaster
)

# Copy config file
add_custom_target(deltacast_endoscopy_tool_tracking_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/deltacast_endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "deltacast_endoscopy_tool_tracking.yaml"
  BYPRODUCTS "deltacast_endoscopy_tool_tracking.yaml"
)
add_dependencies(deltacast_endoscopy_tool_tracking deltacast_endoscopy_tool_tracking_tracking_yaml)
