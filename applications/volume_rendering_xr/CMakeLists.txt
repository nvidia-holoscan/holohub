
 # SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
cmake_minimum_required(VERSION 3.20)
project(volume_rendering_xr)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)


# Set a default build type if none was specified
#if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
#  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
#  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
#endif()
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)


find_package(holoscan 2.0 REQUIRED CONFIG
 PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)


add_executable(volume_rendering_xr
  main.cpp
)

target_link_libraries(volume_rendering_xr
  holoscan::core
  holoscan::ops::xr_transform_op
  holoscan::ops::frame_op
  holoscan::ops::volume_loader
  holoscan::ops::volume_renderer
  holoscan::ops::holoviz
)

set_target_properties(volume_rendering_xr
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../lib:/opt/nvidia/holoscan/lib/"
)

# Copy config file to the build tree
add_custom_target(volume_rendering_xr_config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(volume_rendering_xr volume_rendering_xr_config_yaml)

option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(volume_rendering_xr
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_volume_rendering_sample_data/versions/20230816/zip
    DOWNLOAD_NAME holoscan_volume_rendering_sample_data_20230816.zip
    URL_MD5 cf5602f1dba90f2e50b1d4eefbf8081e
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
  add_dependencies(volume_rendering_xr volume_rendering_xr_data)
endif()

# install binaries
install(TARGETS volume_rendering_xr
  COMPONENT holoscan-apps
)


#if(BUILD_TESTING)
# add_subdirectory(testing)
  # Add test
#  add_test(NAME xr_volume_rendering_test
#           COMMAND xr_volume_rendering --count=100
#           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
#  set_tests_properties(xr_volume_rendering_test PROPERTIES
#                       PASS_REGULAR_EXPRESSION "Application has finished running.")

#endif()