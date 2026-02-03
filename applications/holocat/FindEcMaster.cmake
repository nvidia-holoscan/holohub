# FindEcMaster.cmake
# Find the acontis EC-Master SDK
#
# This module defines:
#  ECMASTER_FOUND - True if EC-Master SDK is found
#  ECMASTER_INCLUDE_DIRS - Include directories for EC-Master
#  ECMASTER_LIBRARIES - Libraries to link against
#  ECMASTER_VERSION - Version of EC-Master SDK
#
# Environment variables used:
#  ECMASTER_ROOT - Root directory of EC-Master installation

# Handle CMake policy for environment variables
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# Get ECMASTER_ROOT from environment if not set as CMake variable
if(NOT ECMASTER_ROOT AND DEFINED ENV{ECMASTER_ROOT})
    set(ECMASTER_ROOT $ENV{ECMASTER_ROOT})
endif()

# Find include directory
find_path(ECMASTER_INCLUDE_DIR
    NAMES EcMaster.h
    PATHS
        ${ECMASTER_ROOT}/SDK/INC
        /opt/acontis/ecmaster/SDK/INC
        ${CMAKE_SOURCE_DIR}/../ethercat/ecm/SDK/INC
        /usr/local/include/ecmaster
    DOC "EC-Master include directory"
)

set (TEMPARCH ${CMAKE_SYSTEM_PROCESSOR})
# Determine library directory based on architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(TEMPARCH MATCHES "aarch64|arm64|AARCH64|ARM64")
        set(ECMASTER_ARCH "arm64")
    else()
        set(ECMASTER_ARCH "x64")
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7|armv8)")
        # ARM 32-bit
        set(ECMASTER_ARCH "arm")
    else()
        set(ECMASTER_ARCH "x86")
    endif()
endif()

# Find main EC-Master library
find_library(ECMASTER_LIBRARY
    NAMES EcMaster libEcMaster.so
    PATHS
        ${ECMASTER_ROOT}/SDK/LIB/Linux/${ECMASTER_ARCH}
        ${ECMASTER_ROOT}/Bin/Linux/${ECMASTER_ARCH}
        /opt/acontis/ecmaster/lib
        ${CMAKE_SOURCE_DIR}/../ethercat/ecm/Bin/Linux/${ECMASTER_ARCH}
        /usr/local/lib
    DOC "EC-Master library"
)

# Get library directory for finding link layer libraries
if(ECMASTER_LIBRARY)
    get_filename_component(ECMASTER_LIBRARY_DIR ${ECMASTER_LIBRARY} DIRECTORY)
endif()

# Find all available link layer libraries
set(ECMASTER_LINK_LIBRARIES)
set(ECMASTER_LINK_LAYER_NAMES
    emllSockRaw
    emllDpdk
    emllIntelGbe
    emllRTL8169
    emllVlan
    emllRemote
    emllCCAT
    emllBcmNetXtreme
    emllLAN743x
    emllDW3504
    emllAlteraTSE
)

foreach(lib_name IN LISTS ECMASTER_LINK_LAYER_NAMES)
    find_library(ECMASTER_${lib_name}_LIBRARY
        NAMES ${lib_name} lib${lib_name}.so
        PATHS ${ECMASTER_LIBRARY_DIR}
        NO_DEFAULT_PATH
    )
    if(ECMASTER_${lib_name}_LIBRARY)
        list(APPEND ECMASTER_LINK_LIBRARIES ${ECMASTER_${lib_name}_LIBRARY})
        message(STATUS "Found EC-Master link layer: ${lib_name}")
    endif()
endforeach()

# Try to determine version from EcVersion.h
if(ECMASTER_INCLUDE_DIR)
    set(ECVERSION_FILE "${ECMASTER_INCLUDE_DIR}/EcVersion.h")
    if(EXISTS ${ECVERSION_FILE})
        file(READ ${ECVERSION_FILE} ECVERSION_CONTENT)
        string(REGEX MATCH "#define EC_VERSION_MAJ[ \t]+([0-9]+)" _ ${ECVERSION_CONTENT})
        set(ECMASTER_VERSION_MAJOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define EC_VERSION_MIN[ \t]+([0-9]+)" _ ${ECVERSION_CONTENT})
        set(ECMASTER_VERSION_MINOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define EC_VERSION_SERVICEPACK[ \t]+([0-9]+)" _ ${ECVERSION_CONTENT})
        set(ECMASTER_VERSION_PATCH ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define EC_VERSION_BUILD[ \t]+([0-9]+)" _ ${ECVERSION_CONTENT})
        set(ECMASTER_VERSION_BUILD ${CMAKE_MATCH_1})
        
        if(ECMASTER_VERSION_MAJOR AND ECMASTER_VERSION_MINOR AND ECMASTER_VERSION_PATCH)
            if(ECMASTER_VERSION_BUILD)
                set(ECMASTER_VERSION "${ECMASTER_VERSION_MAJOR}.${ECMASTER_VERSION_MINOR}.${ECMASTER_VERSION_PATCH}.${ECMASTER_VERSION_BUILD}")
            else()
                set(ECMASTER_VERSION "${ECMASTER_VERSION_MAJOR}.${ECMASTER_VERSION_MINOR}.${ECMASTER_VERSION_PATCH}")
            endif()
        endif()
    endif()
endif()

# Handle standard CMake find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EcMaster
    REQUIRED_VARS ECMASTER_INCLUDE_DIR ECMASTER_LIBRARY
    VERSION_VAR ECMASTER_VERSION
)

# Set output variables
if(EcMaster_FOUND)
    set(ECMASTER_LIBRARIES ${ECMASTER_LIBRARY} ${ECMASTER_LINK_LIBRARIES})
    set(ECMASTER_INCLUDE_DIRS ${ECMASTER_INCLUDE_DIR})
    
    # Also check for Linux-specific include directory
    if(EXISTS "${ECMASTER_INCLUDE_DIR}/Linux")
        list(APPEND ECMASTER_INCLUDE_DIRS "${ECMASTER_INCLUDE_DIR}/Linux")
    endif()
    
    # Create imported target
    if(NOT TARGET EcMaster::EcMaster)
        add_library(EcMaster::EcMaster SHARED IMPORTED)
        set_target_properties(EcMaster::EcMaster PROPERTIES
            IMPORTED_LOCATION ${ECMASTER_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES "${ECMASTER_INCLUDE_DIRS}"
        )
        
        # Add link layer libraries as dependencies
        if(ECMASTER_LINK_LIBRARIES)
            set_target_properties(EcMaster::EcMaster PROPERTIES
                INTERFACE_LINK_LIBRARIES "${ECMASTER_LINK_LIBRARIES}"
            )
        endif()
    endif()
    
    message(STATUS "Found EC-Master SDK:")
    message(STATUS "  Version: ${ECMASTER_VERSION}")
    message(STATUS "  Include: ${ECMASTER_INCLUDE_DIRS}")
    message(STATUS "  Library: ${ECMASTER_LIBRARY}")
    message(STATUS "  Link layers: ${ECMASTER_LINK_LIBRARIES}")
endif()

# Mark variables as advanced
mark_as_advanced(
    ECMASTER_INCLUDE_DIR
    ECMASTER_LIBRARY
    ECMASTER_LIBRARY_DIR
)

foreach(lib_name IN LISTS ECMASTER_LINK_LAYER_NAMES)
    mark_as_advanced(ECMASTER_${lib_name}_LIBRARY)
endforeach()
