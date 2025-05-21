#  Copyright (c) 2022, DELTACAST.TV.
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
project(gxf_videomaster)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

FetchContent_Declare(
    VideoMasterAPIHelper
    GIT_REPOSITORY  https://github.com/deltacasttv/videomaster-api-helper.git
    GIT_TAG         1.3.0
    SOURCE_DIR      "${CMAKE_CURRENT_LIST_DIR}/api_helper"
)
FetchContent_MakeAvailable(VideoMasterAPIHelper)

# Finds the VideoMaster SDK
find_package(VideoMasterHD REQUIRED 6.25)

# Create library
add_library(gxf_videomaster_lib SHARED
  videomaster_base.hpp
  videomaster_base.cpp
  videomaster_source.hpp
  videomaster_source.cpp
  videomaster_transmitter.hpp
  videomaster_transmitter.cpp
)

target_link_libraries(gxf_videomaster_lib
  PUBLIC
    VideoMasterAPIHelper
    VideoMasterHD::Core
    CUDA::cudart
    CUDA::cuda_driver
    GXF::multimedia
    GXF::std
    holoscan::core
    yaml-cpp
)

# Create extension
add_library(gxf_videomaster SHARED
  videomaster_ext.cpp
)
add_library(holoscan::exts::gxf_videomaster ALIAS gxf_videomaster)

target_link_libraries(gxf_videomaster
  PUBLIC gxf_videomaster_lib
)
