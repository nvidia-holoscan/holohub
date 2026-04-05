# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(atracsys_visualizer CXX)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(atracsys_visualizer
  atracsys_visualizer.cpp
)

target_link_libraries(atracsys_visualizer
  PRIVATE
  holoscan::core
  holoscan::ops::bayer_demosaic
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
  yaml-cpp::yaml-cpp
  atracsys_mode_switcher
)

if(ATRACSYS_VISUALIZER_ENABLE_LIVE_CAMERA)
  target_link_libraries(atracsys_visualizer PRIVATE atracsys_camera)
  target_compile_definitions(atracsys_visualizer PRIVATE ATRACSYS_HAVE_LIVE_CAMERA=1)
endif()

add_custom_target(atracsys_visualizer_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer_test.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/geometry10.ini" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer.yaml" ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer_test.yaml" ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/geometry10.ini" ${CMAKE_BINARY_DIR}
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/atracsys_visualizer_test.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/geometry10.ini"
)
add_dependencies(atracsys_visualizer atracsys_visualizer_deps)

if(BUILD_TESTING)
  add_test(NAME atracsys_visualizer_replay_test
           COMMAND atracsys_visualizer
                   --source replayer
                   --config ${CMAKE_CURRENT_BINARY_DIR}/atracsys_visualizer_test.yaml
                   --data "${HOLOHUB_DATA_DIR}/atracsys_visualizer"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(atracsys_visualizer_replay_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "ERROR;Error;Failed")
endif()

install(
  TARGETS atracsys_visualizer
  DESTINATION bin/atracsys_visualizer/cpp
)

install(
  FILES
    atracsys_visualizer.yaml
    atracsys_visualizer_test.yaml
    geometry10.ini
  DESTINATION bin/atracsys_visualizer/cpp
)
