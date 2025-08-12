# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# FindStreaming.cmake
# Find Streaming libraries and create imported targets
#
# This module finds the StreamSDK libraries and creates imported targets
# for use in applications and other CMake projects.
#
# Variables set by this module:
#   StreamSDK_FOUND - TRUE if StreamSDK is found
#   StreamSDK_INCLUDE_DIRS - Include directories for StreamSDK
#   StreamSDK_LIBRARIES - List of all StreamSDK libraries
#   StreamSDK_LIB_DIR - Directory containing StreamSDK libraries
#
# Imported targets created by this module:
#   StreamSDK::StreamingClient - Main streaming client library
#   StreamSDK::All - All StreamSDK libraries combined

cmake_minimum_required(VERSION 3.20)

# Set the search path for StreamSDK libraries
# This should be relative to the operator directory
set(StreamSDK_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(StreamSDK_LIB_DIR "${StreamSDK_ROOT_DIR}/lib")
set(StreamSDK_INCLUDE_DIR "${StreamSDK_ROOT_DIR}")

# Check if the main library directory exists
if(NOT EXISTS "${StreamSDK_LIB_DIR}")
    set(StreamSDK_FOUND FALSE)
    if(StreamSDK_FIND_REQUIRED)
        message(FATAL_ERROR "StreamSDK library directory not found: ${StreamSDK_LIB_DIR}")
    endif()
    return()
endif()

# Define all potential StreamSDK libraries
set(StreamSDK_LIBRARY_NAMES
    StreamingClient
    StreamClientShared
    NvStreamBase
    Poco
    ssl.3
    SCI
    protobuf.9
    NvStreamServer
    NvStreamingSession
    NvSignalingServer
    NvSessionNegotiator
    NvcfBlasUpload
    NicllsHandlerServer
    messagebus
    InputStreamShared
    cudart.12.0.107
    cudart.12
    crypto.3
    AudioStreamShared
)

# Find each library and create imported targets
set(StreamSDK_LIBRARIES)
set(StreamSDK_FOUND_LIBRARIES)

foreach(LIB_NAME IN LISTS StreamSDK_LIBRARY_NAMES)
    set(LIB_PATH "${StreamSDK_LIB_DIR}/lib${LIB_NAME}.so")
    
    if(EXISTS "${LIB_PATH}")
        # Create imported target for this library
        add_library(StreamSDK::${LIB_NAME} SHARED IMPORTED)
        set_target_properties(StreamSDK::${LIB_NAME} PROPERTIES
            IMPORTED_LOCATION "${LIB_PATH}"
            IMPORTED_NO_SONAME ON
        )
        
        list(APPEND StreamSDK_LIBRARIES "${LIB_PATH}")
        list(APPEND StreamSDK_FOUND_LIBRARIES "StreamSDK::${LIB_NAME}")
        
        message(STATUS "Found StreamSDK library: ${LIB_NAME}")
    else()
        message(STATUS "StreamSDK library not found (optional): ${LIB_NAME}")
    endif()
endforeach()

# Check if we found the core library
if(NOT TARGET StreamSDK::StreamingClient)
    set(StreamSDK_FOUND FALSE)
    if(StreamSDK_FIND_REQUIRED)
        message(FATAL_ERROR "Core StreamSDK library libStreamingClient.so not found in ${StreamSDK_LIB_DIR}")
    endif()
    return()
endif()

# Create a combined target that includes all found libraries
add_library(StreamSDK::All INTERFACE IMPORTED)
set_target_properties(StreamSDK::All PROPERTIES
    INTERFACE_LINK_LIBRARIES "${StreamSDK_FOUND_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${StreamSDK_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
)

# Create main StreamingClient target with proper dependencies
set_target_properties(StreamSDK::StreamingClient PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${StreamSDK_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
)

# Set up additional dependencies for the main target
if(TARGET StreamSDK::StreamClientShared)
    set_target_properties(StreamSDK::StreamingClient PROPERTIES
        INTERFACE_LINK_LIBRARIES "StreamSDK::StreamClientShared"
    )
endif()

# Set variables for compatibility
set(StreamSDK_INCLUDE_DIRS "${StreamSDK_INCLUDE_DIR}")
set(StreamSDK_FOUND TRUE)

# Create symbolic link for libnvmessagebus.so -> libmessagebus.so if messagebus exists
if(TARGET StreamSDK::messagebus)
    set(MESSAGEBUS_LINK "${StreamSDK_LIB_DIR}/libnvmessagebus.so")
    set(MESSAGEBUS_TARGET "${StreamSDK_LIB_DIR}/libmessagebus.so")
    
    if(NOT EXISTS "${MESSAGEBUS_LINK}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink 
                "${MESSAGEBUS_TARGET}" 
                "${MESSAGEBUS_LINK}"
            RESULT_VARIABLE SYMLINK_RESULT
        )
        if(SYMLINK_RESULT EQUAL 0)
            message(STATUS "Created symbolic link: libnvmessagebus.so -> libmessagebus.so")
        endif()
    endif()
endif()

# Print summary
message(STATUS "StreamSDK found: ${StreamSDK_FOUND}")
message(STATUS "StreamSDK include directory: ${StreamSDK_INCLUDE_DIRS}")
message(STATUS "StreamSDK library directory: ${StreamSDK_LIB_DIR}")
message(STATUS "StreamSDK libraries found: ${StreamSDK_FOUND_LIBRARIES}")

# Define helper function to copy StreamSDK libraries to target directory
function(copy_streamsdk_libraries TARGET_NAME DESTINATION_DIR)
    # Create directory
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DESTINATION_DIR}"
    )
    
    # Copy each library
    foreach(LIB_PATH IN LISTS StreamSDK_LIBRARIES)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LIB_PATH}"
                "${DESTINATION_DIR}/"
        )
    endforeach()
    
    # Copy symbolic link if it exists
    if(EXISTS "${StreamSDK_LIB_DIR}/libnvmessagebus.so")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${StreamSDK_LIB_DIR}/libnvmessagebus.so"
                "${DESTINATION_DIR}/"
        )
    endif()
endfunction() 
