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

cmake_minimum_required(VERSION 3.20)
project(holoscan_gstreamer_bridge)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find Holoscan
find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GSTREAMER_CUDA gstreamer-cuda-1.0)  # Optional - only available in GStreamer 1.24+
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# Create the Holoscan GStreamer Bridge library
add_library(holoscan_gstreamer_bridge SHARED
  gst/app_src.cpp
  gst/buffer.cpp
  gst/caps.cpp
  gst/memory.cpp
  gst/message.cpp
  gst/video_info.cpp
  gst_src_bridge.cpp
  gst_src_op.cpp
  gst_src_resource.cpp
  gst_video_recorder_op.cpp
  gst_pipeline_bus_monitor.cpp
)

# Create alias for consistent naming
add_library(holoscan::ops::gstreamer_bridge ALIAS holoscan_gstreamer_bridge)

# Set target properties
set_target_properties(holoscan_gstreamer_bridge PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Set include directories (PUBLIC so they propagate to applications)
target_include_directories(holoscan_gstreamer_bridge
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_BASE_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
)

# Link libraries for custom sink
target_link_libraries(holoscan_gstreamer_bridge
    PUBLIC
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_BASE_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    holoscan::core)

# Set compile flags (PRIVATE since they're compiler-specific)
target_compile_options(holoscan_gstreamer_bridge PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_BASE_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER})

# Configure CUDA support (optional - requires GStreamer 1.24+)
if(GSTREAMER_CUDA_FOUND)
    message(STATUS "GStreamer CUDA support: ENABLED (found gstreamer-cuda-1.0)")

    # Add CUDA include directories
    target_include_directories(holoscan_gstreamer_bridge
        PUBLIC
        ${GSTREAMER_CUDA_INCLUDE_DIRS}
    )

    # Add CUDA libraries
    target_link_libraries(holoscan_gstreamer_bridge
        PUBLIC
        ${GSTREAMER_CUDA_LIBRARIES}
    )

    # Add CUDA compile flags
    target_compile_options(holoscan_gstreamer_bridge PRIVATE
        ${GSTREAMER_CUDA_CFLAGS_OTHER}
    )

    # Suppress unstable API warnings for CUDA
    target_compile_definitions(holoscan_gstreamer_bridge PRIVATE
        GST_USE_UNSTABLE_API
    )
else()
    message(STATUS "GStreamer CUDA support: DISABLED (gstreamer-cuda-1.0 not found - requires GStreamer 1.24+)")
endif()

# Install library
install(TARGETS holoscan_gstreamer_bridge DESTINATION lib)

# Install headers
install(FILES
  gst/allocator.hpp
  gst/app_src.hpp
  gst/buffer.hpp
  gst/bus.hpp
  gst/caps.hpp
  gst/config.hpp
  gst/cuda_context.hpp
  gst/element.hpp
  gst/error.hpp
  gst/memory.hpp
  gst/message.hpp
  gst/mini_object.hpp
  gst/object.hpp
  gst/pad.hpp
  gst/pipeline.hpp
  gst/video_info.hpp
  gst_src_bridge.hpp
  gst_src_op.hpp
  gst_src_resource.hpp
  gst_video_recorder_op.hpp
  gst_pipeline_bus_monitor.hpp
  DESTINATION include/holoscan/operators/gstreamer)

# Export the library for use by other CMake projects
export(TARGETS holoscan_gstreamer_bridge
       FILE "${CMAKE_CURRENT_BINARY_DIR}/holoscan_gstreamer_bridge-targets.cmake")

# Print some information
message(STATUS "Holoscan GStreamer Sink Library")
message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
message(STATUS "GStreamer library dirs: ${GSTREAMER_LIBRARY_DIRS}")
