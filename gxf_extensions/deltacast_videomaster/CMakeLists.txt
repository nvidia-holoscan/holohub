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
project(gxf_videomaster)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Finds the VideoMaster SDK
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
find_package(VideoMaster REQUIRED)

# Create library
add_library(gxf_videomaster_lib SHARED
  videomaster_base.hpp
  videomaster_base.cpp
  videomaster_source.hpp
  videomaster_source.cpp
  videomaster_transmitter.hpp
  videomaster_transmitter.cpp
  video_information/video_information.hpp
  video_information/sdi_video_information.hpp
  video_information/sdi_video_information.cpp
  video_information/dv_video_information.hpp
  video_information/dv_video_information.cpp
)

target_link_libraries(gxf_videomaster_lib
  PUBLIC
    VideoMaster::videomasterhd_core
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
