# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# FindStreaming.cmake
# Find Streaming libraries and create imported targets
#
# This module finds the Streaming libraries and creates imported targets
# for use in applications and other CMake projects.
#
# Variables set by this module:
#   Streaming_FOUND - TRUE if Streaming is found
#   Streaming_INCLUDE_DIRS - Include directories for Streaming
#   Streaming_LIBRARIES - List of all Streaming libraries
#   Streaming_LIB_DIR - Directory containing Streaming libraries
#
# Imported targets created by this module:
#   Streaming::StreamingClient - Main streaming client library
#   Streaming::All - All Streaming libraries combined

cmake_minimum_required(VERSION 3.20)

# Include FindPackageHandleStandardArgs for standard CMake behavior
include(FindPackageHandleStandardArgs)

# Set the search path for Streaming libraries
# This should be relative to the operator directory
set(Streaming_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(Streaming_LIB_DIR "${Streaming_ROOT_DIR}/lib")
set(Streaming_INCLUDE_DIR "${Streaming_ROOT_DIR}")

# Check if the main library directory exists
if(NOT EXISTS "${Streaming_LIB_DIR}")
    set(Streaming_FOUND FALSE)
    if(Streaming_FIND_REQUIRED)
        message(FATAL_ERROR "Streaming library directory not found: ${Streaming_LIB_DIR}")
    endif()
    return()
endif()

# Define all potential Streaming libraries
set(Streaming_LIBRARY_NAMES
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
set(Streaming_LIBRARIES)
set(Streaming_FOUND_LIBRARIES)

foreach(LIB_NAME IN LISTS Streaming_LIBRARY_NAMES)
    set(LIB_PATH "${Streaming_LIB_DIR}/lib${LIB_NAME}.so")

    if(EXISTS "${LIB_PATH}")
        # Create imported target for this library
        add_library(Streaming::${LIB_NAME} SHARED IMPORTED)
        set_target_properties(Streaming::${LIB_NAME} PROPERTIES
            IMPORTED_LOCATION "${LIB_PATH}"
            IMPORTED_NO_SONAME ON
        )

        list(APPEND Streaming_LIBRARIES "${LIB_PATH}")
        list(APPEND Streaming_FOUND_LIBRARIES "Streaming::${LIB_NAME}")

        message(STATUS "Found Streaming library: ${LIB_NAME}")
    else()
        message(STATUS "Streaming library not found (optional): ${LIB_NAME}")
    endif()
endforeach()

# Find the core StreamingClient library
find_library(Streaming_STREAMING_CLIENT_LIBRARY
    NAMES StreamingClient
    PATHS "${Streaming_LIB_DIR}"
    NO_DEFAULT_PATH
)

# Check if we found the core library and set up variables
if(Streaming_STREAMING_CLIENT_LIBRARY AND TARGET Streaming::StreamingClient)
    set(Streaming_CORE_FOUND TRUE)

    # Create a combined target that includes all found libraries
    add_library(Streaming::All INTERFACE IMPORTED)
    set_target_properties(Streaming::All PROPERTIES
        INTERFACE_LINK_LIBRARIES "${Streaming_FOUND_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Streaming_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Create main StreamingClient target with proper dependencies
    set_target_properties(Streaming::StreamingClient PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Streaming_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Set up additional dependencies for the main target
    if(TARGET Streaming::StreamClientShared)
        set_target_properties(Streaming::StreamingClient PROPERTIES
            INTERFACE_LINK_LIBRARIES "Streaming::StreamClientShared"
        )
    endif()
else()
    set(Streaming_CORE_FOUND FALSE)
endif()

# Set variables for compatibility
set(Streaming_INCLUDE_DIRS "${Streaming_INCLUDE_DIR}")

# Use FindPackageHandleStandardArgs to handle the standard CMake find behavior
find_package_handle_standard_args(Streaming
    FOUND_VAR Streaming_FOUND
    REQUIRED_VARS Streaming_STREAMING_CLIENT_LIBRARY Streaming_INCLUDE_DIR Streaming_LIB_DIR
    VERSION_VAR "1.0"
)

# Only proceed if the package was found
if(Streaming_FOUND)
    # Create symbolic link for libnvmessagebus.so -> libmessagebus.so if messagebus exists
    if(TARGET Streaming::messagebus)
        set(MESSAGEBUS_LINK "${Streaming_LIB_DIR}/libnvmessagebus.so")
        set(MESSAGEBUS_TARGET "${Streaming_LIB_DIR}/libmessagebus.so")

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
    message(STATUS "Streaming found: ${Streaming_FOUND}")
    message(STATUS "Streaming include directory: ${Streaming_INCLUDE_DIRS}")
    message(STATUS "Streaming library directory: ${Streaming_LIB_DIR}")
    message(STATUS "Streaming libraries found: ${Streaming_FOUND_LIBRARIES}")
endif()

# Define helper function to copy Streaming libraries to target directory
function(copy_streaming_libraries TARGET_NAME DESTINATION_DIR)
    # Create directory
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DESTINATION_DIR}"
    )

    # Copy each library
    foreach(LIB_PATH IN LISTS Streaming_LIBRARIES)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LIB_PATH}"
                "${DESTINATION_DIR}/"
        )
    endforeach()

    # Copy symbolic link if it exists
    if(EXISTS "${Streaming_LIB_DIR}/libnvmessagebus.so")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${Streaming_LIB_DIR}/libnvmessagebus.so"
                "${DESTINATION_DIR}/"
        )
    endif()
endfunction()
