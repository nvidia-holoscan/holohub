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
project(holocat LANGUAGES CXX)

# Find required packages
find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find EC-Master SDK (REQUIRED - no fallback)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
find_package(EcMaster REQUIRED)

message(STATUS "HoloCat: EC-Master SDK found:")
message(STATUS "  Include: ${ECMASTER_INCLUDE_DIRS}")
message(STATUS "  Libraries: ${ECMASTER_LIBRARIES}")

# Source files (no conditional compilation)
set(HOLOCAT_SOURCES
    main.cpp
    holocat_op.cpp
    holocat_app.cpp
    hc_data_tx_op.cpp
    hc_data_rx_op.cpp
)

set(HOLOCAT_HEADERS
    holocat_op.hpp
    holocat_app.hpp
    hc_data_tx_op.hpp
    hc_data_rx_op.hpp
)

# Create executable
add_executable(holocat ${HOLOCAT_SOURCES} ${HOLOCAT_HEADERS})

# Include directories
target_include_directories(holocat PRIVATE 
    .
    ${ECMASTER_INCLUDE_DIRS}
)

# Compile options
target_compile_options(holocat PRIVATE 
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-packed-not-aligned
)

# Link libraries (EtherCAT always linked)
target_link_libraries(holocat 
    holoscan::core 
    ${ECMASTER_LIBRARIES}
    pthread
    rt
)

# Set RPATH for runtime library discovery and output directory
set_target_properties(holocat PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/.."
)

# Optionally set capabilities for real-time operation
# Set HOLOCAT_SET_CAPABILITIES=ON to enable this during build
# Otherwise, set capabilities manually after build:
#   sudo setcap 'cap_net_raw,cap_sys_nice,cap_ipc_lock=ep' holocat

option(HOLOCAT_SET_CAPABILITIES "Set Linux capabilities during build" OFF)
if(HOLOCAT_SET_CAPABILITIES)
    message(WARNING "Setting capabilities for real-time EtherCAT operation - requires sudo privileges - ensure you have appropriate access")
    add_custom_command(TARGET holocat POST_BUILD
        COMMAND sudo setcap 'cap_net_raw,cap_sys_nice,cap_ipc_lock=ep' $<TARGET_FILE:holocat>
        COMMENT "Setting capabilities for real-time EtherCAT operation"
    )
endif()
