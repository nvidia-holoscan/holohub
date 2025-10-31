# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FindStreamingClient.cmake
# Find StreamingClient libraries and create imported targets
#
# This module finds the StreamingClient libraries and creates imported targets
# for use in applications and other CMake projects.
#
# Variables set by this module:
#   StreamingClient_FOUND - TRUE if StreamingClient is found
#   StreamingClient_INCLUDE_DIRS - Include directories for StreamingClient
#   StreamingClient_LIBRARIES - List of all StreamingClient libraries
#   StreamingClient_LIB_DIR - Directory containing StreamingClient libraries
#
# Imported targets created by this module:
#   StreamingClient::StreamingClient - Main streaming client library
#   StreamingClient::All - All StreamingClient libraries combined

cmake_minimum_required(VERSION 3.20)

# Include FindPackageHandleStandardArgs for standard CMake behavior
include(FindPackageHandleStandardArgs)

# Set the search path for StreamingClient libraries
# This should be relative to the operator directory
set(StreamingClient_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(StreamingClient_STREAMINGCLIENT_DIR "${StreamingClient_ROOT_DIR}/holoscan_client_cloud_streaming")

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH_DIR "aarch64")
else()
    set(ARCH_DIR "x86_64") # Default to x86_64
endif()

set(StreamingClient_LIB_DIR "${StreamingClient_STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}")
set(StreamingClient_INCLUDE_DIR "${StreamingClient_STREAMINGCLIENT_DIR}/include")

# Check if the main library directory exists
if(NOT EXISTS "${StreamingClient_LIB_DIR}")
    set(StreamingClient_FOUND FALSE)
    if(StreamingClient_FIND_REQUIRED)
        message(FATAL_ERROR "StreamingClient library directory not found: ${StreamingClient_LIB_DIR}")
    endif()
    return()
endif()

# Check if the include directory exists
if(NOT EXISTS "${StreamingClient_INCLUDE_DIR}")
    set(StreamingClient_FOUND FALSE)
    if(StreamingClient_FIND_REQUIRED)
        message(FATAL_ERROR "StreamingClient include directory not found: ${StreamingClient_INCLUDE_DIR}")
    endif()
    return()
endif()

# Define all potential StreamingClient libraries
set(StreamingClient_LIBRARY_NAMES
    StreamingClient
    StreamClientShared
    NvStreamBase
    NvStreamingSession
    Poco
    ssl.3
    crypto.3
    cudart.12
)

# Initialize lists
set(StreamingClient_FOUND_LIBRARIES)
set(StreamingClient_LIBRARIES)

# Find each library and create imported targets
foreach(lib_name ${StreamingClient_LIBRARY_NAMES})
    # Find the library
    find_library(StreamingClient_${lib_name}_LIBRARY
        NAMES ${lib_name} lib${lib_name}
        PATHS ${StreamingClient_LIB_DIR}
        NO_DEFAULT_PATH
    )

    if(StreamingClient_${lib_name}_LIBRARY)
        # Add to found libraries list
        list(APPEND StreamingClient_FOUND_LIBRARIES ${StreamingClient_${lib_name}_LIBRARY})
        list(APPEND StreamingClient_LIBRARIES ${StreamingClient_${lib_name}_LIBRARY})

        # Create imported target for this library
        add_library(StreamingClient::${lib_name} SHARED IMPORTED)
        set_target_properties(StreamingClient::${lib_name} PROPERTIES
            IMPORTED_LOCATION "${StreamingClient_${lib_name}_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${StreamingClient_INCLUDE_DIR}"
        )

        message(STATUS "Found StreamingClient library: ${lib_name} at ${StreamingClient_${lib_name}_LIBRARY}")
    else()
        message(STATUS "StreamingClient library not found: ${lib_name}")
    endif()
endforeach()

# Find the main StreamingClient library (required)
find_library(StreamingClient_MAIN_LIBRARY
    NAMES StreamingClient libStreamingClient
    PATHS ${StreamingClient_LIB_DIR}
    NO_DEFAULT_PATH
)

# Check if we found the core library and set up variables
if(StreamingClient_MAIN_LIBRARY AND TARGET StreamingClient::StreamingClient)
    set(StreamingClient_CORE_FOUND TRUE)

    # Create a combined target that includes all found libraries
    add_library(StreamingClient::All INTERFACE IMPORTED)
    set_target_properties(StreamingClient::All PROPERTIES
        INTERFACE_LINK_LIBRARIES "${StreamingClient_FOUND_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${StreamingClient_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Create main StreamingClient target with proper dependencies
    set_target_properties(StreamingClient::StreamingClient PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${StreamingClient_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1"
    )

    # Set up additional dependencies for the main target
    if(TARGET StreamingClient::StreamClientShared)
        set_target_properties(StreamingClient::StreamingClient PROPERTIES
            INTERFACE_LINK_LIBRARIES "StreamingClient::StreamClientShared"
        )
    endif()

    # Add other dependencies if found
    set(StreamingClient_DEPENDENCIES)
    foreach(dep_lib NvStreamBase NvStreamingSession Poco)
        if(TARGET StreamingClient::${dep_lib})
            list(APPEND StreamingClient_DEPENDENCIES StreamingClient::${dep_lib})
        endif()
    endforeach()

    if(StreamingClient_DEPENDENCIES)
        set_property(TARGET StreamingClient::StreamingClient APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES ${StreamingClient_DEPENDENCIES}
        )
    endif()
else()
    set(StreamingClient_CORE_FOUND FALSE)
endif()

# Set variables for compatibility
set(StreamingClient_INCLUDE_DIRS "${StreamingClient_INCLUDE_DIR}")

# Use FindPackageHandleStandardArgs to handle the standard CMake find behavior
find_package_handle_standard_args(StreamingClient
    FOUND_VAR StreamingClient_FOUND
    REQUIRED_VARS StreamingClient_MAIN_LIBRARY StreamingClient_INCLUDE_DIR StreamingClient_LIB_DIR
    VERSION_VAR "1.0"
)

# Only proceed if the package was found
if(StreamingClient_FOUND)
    message(STATUS "StreamingClient package found successfully")
    message(STATUS "  Include directory: ${StreamingClient_INCLUDE_DIR}")
    message(STATUS "  Library directory: ${StreamingClient_LIB_DIR}")
    message(STATUS "  Found libraries: ${StreamingClient_FOUND_LIBRARIES}")
else()
    message(STATUS "StreamingClient package not found")
endif()

# Mark variables as advanced
mark_as_advanced(
    StreamingClient_ROOT_DIR
    StreamingClient_LIB_DIR
    StreamingClient_INCLUDE_DIR
    StreamingClient_MAIN_LIBRARY
)
