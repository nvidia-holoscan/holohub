# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(ARGUS_METADATA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
holoscan_add_flatbuffer_schema(
    TARGET argus_frame_metadata_generated
    SCHEMAS argus_frame_metadata.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${ARGUS_METADATA_GENERATED_DIR}/argus_capture_op"
    BFBS_EMBED
    TRAITS_HEADER argus_frame_metadata_schema_traits.hpp
    COMPATIBILITY_EPOCH 1
    MAX_SERIALIZED_SIZE 4096
)

# Compile the actual operator once. Unit tests link these same objects to a deterministic capture
# backend so lifecycle and publication can be exercised without Jetson drivers or a camera.
add_library(argus_capture_impl OBJECT argus_capture_op.cpp)
set_target_properties(argus_capture_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(argus_capture_impl argus_frame_metadata_generated)
target_compile_features(argus_capture_impl PUBLIC cxx_std_20)
target_include_directories(argus_capture_impl PUBLIC
    "${PROJECT_SOURCE_DIR}/operators" "${ARGUS_METADATA_GENERATED_DIR}")
target_link_libraries(argus_capture_impl PUBLIC holoscan::core holoscan::sensor_io Threads::Threads)

add_library(argus_capture_op SHARED
    $<TARGET_OBJECTS:argus_capture_impl> argus_capture_service.cpp)
add_library(holoscan::argus_capture_op ALIAS argus_capture_op)
target_compile_features(argus_capture_op PUBLIC cxx_std_20)
target_include_directories(argus_capture_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<BUILD_INTERFACE:${ARGUS_METADATA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(argus_capture_op PUBLIC holoscan::core holoscan::sensor_io
    PRIVATE Threads::Threads)

find_path(ARGUS_INCLUDE_DIR Argus/Argus.h
    HINTS /usr/src/jetson_multimedia_api/argus/include /usr/src/jetson_multimedia_api/include
    REQUIRED)
find_path(NVBUFSURFACE_INCLUDE_DIR nvbufsurface.h
    HINTS /usr/src/jetson_multimedia_api/include REQUIRED)
find_path(ARGUS_EGL_INCLUDE_DIR EGL/egl.h REQUIRED)
find_library(ARGUS_LIBRARY NAMES nvargus_socketclient
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra REQUIRED)
find_library(NVBUFSURFACE_LIBRARY NAMES nvbufsurface
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra REQUIRED)
# Standard system libEGL; ARGUS_ scopes the cache override to this backend.
# Capture needs EGLDisplay initialization for Argus/CUDA EGLImage buffer sharing,
# including headless operation, independently of downstream visualization.
find_library(ARGUS_EGL_LIBRARY NAMES EGL REQUIRED)
target_include_directories(argus_capture_op PRIVATE
    "${ARGUS_INCLUDE_DIR}" "${NVBUFSURFACE_INCLUDE_DIR}" "${ARGUS_EGL_INCLUDE_DIR}")
target_link_libraries(argus_capture_op PRIVATE CUDA::cuda_driver
    "${ARGUS_LIBRARY}" "${NVBUFSURFACE_LIBRARY}" "${ARGUS_EGL_LIBRARY}")
# The container provides direct link libraries, but their transitive BSP dependencies
# are available only on Jetson. Preserve those direct ELF dependencies and allow unresolved
# references inside the shared libraries. Consumers need the same executable-link policy.
target_link_options(argus_capture_op PUBLIC "LINKER:--allow-shlib-undefined")

install(TARGETS argus_capture_op EXPORT holoscan_camera_targets)
install(FILES argus_capture_op.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/argus_capture_op)
get_target_property(_argus_generated_headers argus_frame_metadata_generated
    HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)
install(FILES ${_argus_generated_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/argus_capture_op)
install(FILES argus_frame_metadata.fbs DESTINATION ${CMAKE_INSTALL_DATADIR}/holoscan-camera/schemas)
if(HOLOSCAN_CAMERA_BUILD_TESTING)
    add_subdirectory(tests)
endif()
