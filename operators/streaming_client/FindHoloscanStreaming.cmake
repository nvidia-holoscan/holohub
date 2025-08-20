# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# FindHoloscanStreaming.cmake
# Find HoloscanStreaming libraries and create imported targets
#
# This module finds the HoloscanStreaming libraries and creates imported targets
# for use in applications and other CMake projects.
#
# Variables set by this module:
#   HoloscanStreaming_FOUND - TRUE if HoloscanStreaming is found
#   HoloscanStreaming_INCLUDE_DIRS - Include directories for HoloscanStreaming
#   HoloscanStreaming_LIBRARIES - List of all HoloscanStreaming libraries
#   HoloscanStreaming_LIB_DIR - Directory containing HoloscanStreaming libraries
#
# Imported targets created by this module:
#   HoloscanStreaming::StreamingClient - Main streaming client library
#   HoloscanStreaming::All - All HoloscanStreaming libraries combined

cmake_minimum_required(VERSION 3.20)

# Include FindPackageHandleStandardArgs for standard CMake behavior
include(FindPackageHandleStandardArgs)

# Set the search path for HoloscanStreaming libraries
# This should be relative to the operator directory
set(HoloscanStreaming_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(HoloscanStreaming_LIB_DIR "${HoloscanStreaming_ROOT_DIR}/lib")
set(HoloscanStreaming_INCLUDE_DIR "${HoloscanStreaming_ROOT_DIR}")

# Check if the main library directory exists
if(NOT EXISTS "${HoloscanStreaming_LIB_DIR}")
    set(HoloscanStreaming_FOUND FALSE)
    if(HoloscanStreaming_FIND_REQUIRED)
        message(FATAL_ERROR "HoloscanStreaming library directory not found: ${HoloscanStreaming_LIB_DIR}")
    endif()
    return()
endif()

# Define all potential HoloscanStreaming libraries
set(HoloscanStreaming_LIBRARY_NAMES
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
set(HoloscanStreaming_LIBRARIES)
set(HoloscanStreaming_FOUND_LIBRARIES)

foreach(LIB_NAME IN LISTS HoloscanStreaming_LIBRARY_NAMES)
    set(LIB_PATH "${HoloscanStreaming_LIB_DIR}/lib${LIB_NAME}.so")

    if(EXISTS "${LIB_PATH}")
        # Create imported target for this library
        add_library(HoloscanStreaming::${LIB_NAME} SHARED IMPORTED)
        set_target_properties(HoloscanStreaming::${LIB_NAME} PROPERTIES
            IMPORTED_LOCATION "${LIB_PATH}"
            IMPORTED_NO_SONAME ON
        )

        list(APPEND HoloscanStreaming_LIBRARIES "${LIB_PATH}")
        list(APPEND HoloscanStreaming_FOUND_LIBRARIES "HoloscanStreaming::${LIB_NAME}")

        message(STATUS "Found HoloscanStreaming library: ${LIB_NAME}")
    else()
        message(STATUS "HoloscanStreaming library not found (optional): ${LIB_NAME}")
    endif()
endforeach()

# Find the core StreamingClient library
find_library(HoloscanStreaming_STREAMING_CLIENT_LIBRARY
    NAMES StreamingClient
    PATHS "${HoloscanStreaming_LIB_DIR}"
    NO_DEFAULT_PATH
)

# Check if we found the core library and set up variables
if(HoloscanStreaming_STREAMING_CLIENT_LIBRARY AND TARGET HoloscanStreaming::StreamingClient)
    set(HoloscanStreaming_CORE_FOUND TRUE)

    # Create a combined target that includes all found libraries
    add_library(HoloscanStreaming::All INTERFACE IMPORTED)
    set_target_properties(HoloscanStreaming::All PROPERTIES
        INTERFACE_LINK_LIBRARIES "${HoloscanStreaming_FOUND_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${HoloscanStreaming_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Create main StreamingClient target with proper dependencies
    set_target_properties(HoloscanStreaming::StreamingClient PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HoloscanStreaming_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Set up additional dependencies for the main target
    if(TARGET HoloscanStreaming::StreamClientShared)
        set_target_properties(HoloscanStreaming::StreamingClient PROPERTIES
            INTERFACE_LINK_LIBRARIES "HoloscanStreaming::StreamClientShared"
        )
    endif()
else()
    set(HoloscanStreaming_CORE_FOUND FALSE)
endif()

# Set variables for compatibility
set(HoloscanStreaming_INCLUDE_DIRS "${HoloscanStreaming_INCLUDE_DIR}")

# Use FindPackageHandleStandardArgs to handle the standard CMake find behavior
find_package_handle_standard_args(HoloscanStreaming
    FOUND_VAR HoloscanStreaming_FOUND
    REQUIRED_VARS HoloscanStreaming_STREAMING_CLIENT_LIBRARY HoloscanStreaming_INCLUDE_DIR HoloscanStreaming_LIB_DIR
    VERSION_VAR "1.0"
)

# Only proceed if the package was found
if(HoloscanStreaming_FOUND)
    # Create symbolic link for libnvmessagebus.so -> libmessagebus.so if messagebus exists
    if(TARGET HoloscanStreaming::messagebus)
        set(MESSAGEBUS_LINK "${HoloscanStreaming_LIB_DIR}/libnvmessagebus.so")
        set(MESSAGEBUS_TARGET "${HoloscanStreaming_LIB_DIR}/libmessagebus.so")

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
    message(STATUS "HoloscanStreaming found: ${HoloscanStreaming_FOUND}")
    message(STATUS "HoloscanStreaming include directory: ${HoloscanStreaming_INCLUDE_DIRS}")
    message(STATUS "HoloscanStreaming library directory: ${HoloscanStreaming_LIB_DIR}")
    message(STATUS "HoloscanStreaming libraries found: ${HoloscanStreaming_FOUND_LIBRARIES}")
endif()

# Define helper function to copy HoloscanStreaming libraries to target directory
function(copy_holoscan_streaming_libraries TARGET_NAME DESTINATION_DIR)
    # Create directory
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DESTINATION_DIR}"
    )

    # Copy each library
    foreach(LIB_PATH IN LISTS HoloscanStreaming_LIBRARIES)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LIB_PATH}"
                "${DESTINATION_DIR}/"
        )
    endforeach()

    # Copy symbolic link if it exists
    if(EXISTS "${HoloscanStreaming_LIB_DIR}/libnvmessagebus.so")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${HoloscanStreaming_LIB_DIR}/libnvmessagebus.so"
                "${DESTINATION_DIR}/"
        )
    endif()
endfunction()
