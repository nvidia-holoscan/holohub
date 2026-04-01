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

# cuDisp Test Suite

find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED libdrm)

# --- Functional: Pure DRM test ---
add_executable(drm_basic_test
    functional/drm_basic_test.cpp
)
target_include_directories(drm_basic_test PRIVATE ${DRM_INCLUDE_DIRS})
target_link_libraries(drm_basic_test PRIVATE ${DRM_LIBRARIES})
target_compile_options(drm_basic_test PRIVATE -Wall -Wextra)
set_target_properties(drm_basic_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Functional: GBM + DRM test ---
pkg_check_modules(GBM REQUIRED gbm)

add_executable(gbm_drm_test
    functional/gbm_drm_test.cpp
)
target_include_directories(gbm_drm_test PRIVATE ${DRM_INCLUDE_DIRS} ${GBM_INCLUDE_DIRS})
target_link_libraries(gbm_drm_test PRIVATE ${DRM_LIBRARIES} ${GBM_LIBRARIES})
target_compile_options(gbm_drm_test PRIVATE -Wall -Wextra)
set_target_properties(gbm_drm_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Functional: GBM -> Vulkan -> CUDA pipeline test ---
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(buffer_pipeline_test
    functional/buffer_pipeline_test.cpp
)
target_include_directories(buffer_pipeline_test PRIVATE
    ${DRM_INCLUDE_DIRS} ${GBM_INCLUDE_DIRS})
target_link_libraries(buffer_pipeline_test PRIVATE
    ${DRM_LIBRARIES} ${GBM_LIBRARIES}
    Vulkan::Vulkan CUDA::cuda_driver)
target_compile_options(buffer_pipeline_test PRIVATE -Wall -Wextra)
set_target_properties(buffer_pipeline_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Functional: DRM format test ---
add_executable(format_test
    functional/format_test.cpp
)
target_include_directories(format_test PRIVATE ${DRM_INCLUDE_DIRS})
target_link_libraries(format_test PRIVATE ${DRM_LIBRARIES})
target_compile_options(format_test PRIVATE -Wall -Wextra)
set_target_properties(format_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- API validation tests (GTest) ---
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(api_validation_test
    api/api_validation_test.cpp
)
target_include_directories(api_validation_test PRIVATE ../include ${DRM_INCLUDE_DIRS})
target_link_libraries(api_validation_test PRIVATE
    cuDisp CUDA::cuda_driver ${DRM_LIBRARIES} GTest::gtest)
target_compile_options(api_validation_test PRIVATE -Wall -Wextra)
set_target_properties(api_validation_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Integration: Host present test ---
add_executable(host_present_test
    integration/host_present_test.cpp
    integration/host_present_kernels.cu
)
target_include_directories(host_present_test PRIVATE
    ../include ${DRM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/integration)
target_link_libraries(host_present_test PRIVATE cuDisp CUDA::cuda_driver ${DRM_LIBRARIES})
set_target_properties(host_present_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Integration: Performance test ---
add_executable(perf_test
    integration/perf_test.cpp
    integration/perf_test_kernels.cu
)
target_include_directories(perf_test PRIVATE
    ../include ${DRM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/integration)
target_link_libraries(perf_test PRIVATE cuDisp CUDA::cuda_driver ${DRM_LIBRARIES})
set_target_properties(perf_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# --- Demo: Standalone cuDisp demo (GPU present) ---
add_executable(cudisp_demo
    demo/cudisp_demo.cpp
    demo/cudisp_demo_kernels.cu
)
target_include_directories(cudisp_demo PRIVATE
    ../include ${DRM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/demo)
target_link_libraries(cudisp_demo PRIVATE cuDisp CUDA::cuda_driver ${DRM_LIBRARIES})
set_target_properties(cudisp_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
