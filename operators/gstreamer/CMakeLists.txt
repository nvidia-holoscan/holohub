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

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find Holoscan
find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GSTREAMER_CUDA REQUIRED gstreamer-cuda-1.0)

# Add compiler definitions
add_definitions(${GSTREAMER_CFLAGS_OTHER})
add_definitions(${GSTREAMER_BASE_CFLAGS_OTHER})
add_definitions(${GSTREAMER_VIDEO_CFLAGS_OTHER})
add_definitions(${GSTREAMER_CUDA_CFLAGS_OTHER})

# Create custom sink library
add_library(holoscan_gstreamer_sink SHARED
  gst/guards.cpp
  gst/iterator.cpp
  gst/buffer.cpp
  gst/caps.cpp
  gst/video_info.cpp
  gst/audio_info.cpp
  gst/map_info.cpp
  gst/mapped_buffer.cpp
  gst_sink_resource.cpp
  gst_sink_operator.cpp
  gst_src_resource.cpp
  gst_src_operator.cpp
)

# Set library directories
link_directories(${GSTREAMER_LIBRARY_DIRS})
link_directories(${GSTREAMER_BASE_LIBRARY_DIRS})
link_directories(${GSTREAMER_VIDEO_LIBRARY_DIRS})
link_directories(${GSTREAMER_CUDA_LIBRARY_DIRS})

# Set include directories (PUBLIC so they propagate to applications)
target_include_directories(holoscan_gstreamer_sink
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_BASE_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${GSTREAMER_CUDA_INCLUDE_DIRS}
)

# Link libraries for custom sink
target_link_libraries(holoscan_gstreamer_sink 
    PUBLIC
    ${GSTREAMER_LIBRARIES} 
    ${GSTREAMER_BASE_LIBRARIES} 
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${GSTREAMER_CUDA_LIBRARIES}
    holoscan::core)

# Set compile flags for custom sink
target_compile_options(holoscan_gstreamer_sink PRIVATE 
    ${GSTREAMER_CFLAGS_OTHER} 
    ${GSTREAMER_BASE_CFLAGS_OTHER} 
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    ${GSTREAMER_CUDA_CFLAGS_OTHER})

# Platform-specific settings
if(APPLE)
    # For macOS, we might need additional frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COCOA_FRAMEWORK Cocoa)
    if(FOUNDATION_FRAMEWORK AND COCOA_FRAMEWORK)
        target_link_libraries(holoscan_gstreamer_sink ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
    endif()
endif()

# Install library
install(TARGETS holoscan_gstreamer_sink DESTINATION lib)

# Install headers
install(FILES 
    gst/guards.hpp
    gst/iterator.hpp
    gst/buffer.hpp
    gst/caps.hpp
    gst/video_info.hpp
    gst/audio_info.hpp
    gst/map_info.hpp
    gst/mapped_buffer.hpp
    gst_sink_resource.hpp
    gst_sink_operator.hpp
    gst_src_resource.hpp
    gst_src_operator.hpp
    core/initializer.hpp
    DESTINATION include/holoscan/operators/gstreamer)

# Export the library for use by other CMake projects
export(TARGETS holoscan_gstreamer_sink 
       FILE "${CMAKE_CURRENT_BINARY_DIR}/holoscan_gstreamer_sink-targets.cmake")

# Print some information
message(STATUS "Holoscan GStreamer Sink Library")
message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
message(STATUS "GStreamer library dirs: ${GSTREAMER_LIBRARY_DIRS}")
message(STATUS "GStreamer CUDA libraries: ${GSTREAMER_CUDA_LIBRARIES}")
