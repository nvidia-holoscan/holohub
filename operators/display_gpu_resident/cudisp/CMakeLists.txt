# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# cuDisp - DRM Display Library with CUDA Integration

enable_language(CUDA)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED libdrm)
pkg_check_modules(GBM REQUIRED gbm)
pkg_check_modules(VULKAN REQUIRED vulkan)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_library(cuDisp SHARED
    cuDisp.cpp
    cuDispDevice.cu
)

target_include_directories(cuDisp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${DRM_INCLUDE_DIRS}
        ${GBM_INCLUDE_DIRS}
        ${VULKAN_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuDisp
    PUBLIC
        ${DRM_LIBRARIES}
        ${GBM_LIBRARIES}
        ${VULKAN_LIBRARIES}
        CUDA::cuda_driver
)

# Set compile options
target_compile_options(cuDisp PRIVATE -Wall -Wextra)

# Optional: enable benchmarking via CUDISP_BENCH CMake option
option(CUDISP_BENCH "Enable present-thread benchmarking" OFF)
if(CUDISP_BENCH)
    target_compile_definitions(cuDisp PRIVATE CUDISP_BENCH=1)
endif()

# Optional: enable debug prints via CUDISP_DEBUG CMake option
option(CUDISP_DEBUG "Enable verbose debug prints" OFF)
if(CUDISP_DEBUG)
    target_compile_definitions(cuDisp PRIVATE CUDISP_DEBUG=1)
endif()

# Set C++ standard (required for std::atomic) and output library name
set_target_properties(cuDisp PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME cudisp
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install rules
install(TARGETS cuDisp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
        include/cuDisp.h
        include/cuDispDevice.h
    DESTINATION include
)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
