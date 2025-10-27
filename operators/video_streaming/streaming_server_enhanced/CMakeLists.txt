
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(streaming_server_enhanced CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Holoscan package
find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set the StreamingServer directory
set(STREAMINGSERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoscan_server_cloud_streaming)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH_DIR "aarch64")
else()
    set(ARCH_DIR "x86_64") # Default to x86_64
endif()

# Find StreamingServer libraries directly
find_library(StreamingServer_LIBRARY
    NAMES StreamingServer libStreamingServer
    PATHS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(NvStreamServer_LIBRARY
    NAMES NvStreamServer libNvStreamServer
    PATHS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(NvStreamBase_LIBRARY
    NAMES NvStreamBase libNvStreamBase
    PATHS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(NvStreamingSession_LIBRARY
    NAMES NvStreamingSession libNvStreamingSession
    PATHS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(Poco_LIBRARY
    NAMES Poco libPoco
    PATHS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

# Create the main streaming server library with all components
add_library(streaming_server_enhanced
    SHARED
    # Split operators for tensor-based processing
    streaming_server_upstream_op.cpp
    streaming_server_downstream_op.cpp

    # Resource
    streaming_server_resource.cpp

    # Shared debug utilities
    frame_debug_utils.cpp
)

# Set C++ standard and properties
set_target_properties(streaming_server_enhanced PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Set target properties
target_include_directories(streaming_server_enhanced
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${STREAMINGSERVER_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(streaming_server_enhanced
    PUBLIC
    holoscan::core
    holoscan::ops::format_converter
    CUDA::cudart
    ${StreamingServer_LIBRARY}
    ${NvStreamServer_LIBRARY}
    ${NvStreamBase_LIBRARY}
    ${NvStreamingSession_LIBRARY}
    ${Poco_LIBRARY}
)

# Create the main alias that applications will use
add_library(holoscan::ops::streaming_server_enhanced ALIAS streaming_server_enhanced)

# Note: DecodedFrameData library removed since we now work with holoscan::Tensor directly



# Install targets
install(TARGETS streaming_server_enhanced
    EXPORT streaming_server_enhanced_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(FILES
    streaming_server_upstream_op.hpp
    streaming_server_downstream_op.hpp
    streaming_server_resource.hpp
    frame_debug_utils.hpp
    DESTINATION include/operators/streaming_server_enhanced
)

# Explicitly install each required StreamingServer library - CMake will fail if any are missing
# Core StreamingServer library
install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libStreamingServer.so
    DESTINATION lib
)

# NVIDIA Stream Server library
install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libNvStreamServer.so
    DESTINATION lib
)

# NVIDIA Stream Base library
install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libNvStreamBase.so
    DESTINATION lib
)

# NVIDIA Streaming Session library (critical for streaming functionality)
install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libNvStreamingSession.so
    DESTINATION lib
)

# Poco library
install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libPoco.so
    DESTINATION lib
)

# SSL/Crypto libraries (if they exist in the server directory)
if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libssl.so.3)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libssl.so.3
        DESTINATION lib
    )
endif()

if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcrypto.so.3)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcrypto.so.3
        DESTINATION lib
    )
endif()

# CUDA runtime libraries (if they exist in the server directory)
if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so.12.0.107)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so.12.0.107
        DESTINATION lib
    )
endif()

if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so.12)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so.12
        DESTINATION lib
    )
endif()

# Install symlinks if they exist
if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libssl.so)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libssl.so
        DESTINATION lib
    )
endif()

if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcrypto.so)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcrypto.so
        DESTINATION lib
    )
endif()

if(EXISTS ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so)
    install(FILES ${STREAMINGSERVER_DIR}/lib/${ARCH_DIR}/libcudart.so
        DESTINATION lib
    )
endif()



# Export targets
install(EXPORT streaming_server_enhanced_targets
    FILE streaming_server_enhanced_targets.cmake
    NAMESPACE holoscan::ops::
    DESTINATION lib/cmake/holoscan
)

# Add Python bindings if requested
if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
