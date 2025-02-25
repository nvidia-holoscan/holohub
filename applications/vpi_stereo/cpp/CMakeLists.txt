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
project(vpi_stereo CXX CUDA)

find_package(holoscan 2.4 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(vpi 3.2 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)

add_executable(vpi_stereo
  main.cpp
  undistort_rectify.cpp
  split_video.cpp
  heat_map.cpp
  stereo_depth_kernels.cu
  crop.cpp
  vpi_stereo.cpp
)
target_link_libraries(vpi_stereo
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::video_stream_replayer
  CUDA::nppig
  Eigen3::Eigen
  vpi
)

# Copy configuration file to current build directory
add_custom_target(vpi_stereo_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/vpi_stereo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "vpi_stereo.yaml"
    BYPRODUCTS "vpi_stereo.yaml"
)

# Reusing the same stereo sample video and calibration file from the stereo_vision application
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  # defines the stereo_vision_data target
  holoscan_download_data(stereo_vision
    URL nvidia/clara-holoscan/holoscan_stereo_video:20241216
    DOWNLOAD_NAME holoscan_stereo_vision_20241216.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 3840
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
    )
endif()

add_custom_target(vpi_stereo_data
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${HOLOHUB_DATA_DIR}/stereo_vision" "${HOLOHUB_DATA_DIR}/vpi_stereo"
  DEPENDS stereo_vision_data
)

add_dependencies(vpi_stereo vpi_stereo_yaml vpi_stereo_data)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/vpi_stereo.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vpi_stereo_testing.yaml ${CONFIG_FILE})

  add_custom_target(vpi_stereo_test ALL
    DEPENDS vpi_stereo
  )

  # Add test
  add_test(NAME vpi_stereo_test
    COMMAND vpi_stereo
              --config ${CMAKE_CURRENT_BINARY_DIR}/vpi_stereo_testing.yaml
              --data "${HOLOHUB_DATA_DIR}/vpi_stereo"
              --source replayer
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(vpi_stereo_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
