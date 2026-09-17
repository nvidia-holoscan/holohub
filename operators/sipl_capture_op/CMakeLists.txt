# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NvSIPL headers are required. Build inside the holoscan-camera container
# (they are downloaded automatically via L4T_VERSION) or ensure JetPack is
# installed so they are present under /usr/src/jetson_sipl_api.
find_path(NVSIPL_INCLUDE_DIR NvSIPLCamera.hpp
    HINTS
        /usr/src/jetson_sipl_api/sipl/include   # extracted by Dockerfile from SIPL API archive
        /usr/local/include
        /usr/include
    DOC "NvSIPL header directory")

if(NOT NVSIPL_INCLUDE_DIR)
    message(FATAL_ERROR
        "NvSIPL headers not found (NvSIPLCamera.hpp). "
        "Build inside the holoscan-camera container (./holohub build-container downloads them) "
        "or install JetPack so the headers are present under /usr/src/jetson_sipl_api.")
endif()
message(STATUS "NvSIPL found: ${NVSIPL_INCLUDE_DIR}")

include("${CMAKE_CURRENT_SOURCE_DIR}/sipl_frame_metadata_flatbuffers.cmake")

add_library(sipl_capture_op SHARED
    sipl_capture_op.cpp
    sipl_capture_service.cpp)
add_library(holoscan::sipl_capture_op ALIAS sipl_capture_op)
add_dependencies(sipl_capture_op sipl_frame_metadata_generated)

target_compile_features(sipl_capture_op PUBLIC cxx_std_20)

include(GNUInstallDirs)

target_include_directories(sipl_capture_op
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
        $<BUILD_INTERFACE:${SIPL_METADATA_GENERATED_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sipl_capture_op>)

# Detect SIPL API major version from L4T_MAJ_VER, preferring a caller-supplied cache variable
# (-DL4T_MAJ_VER=<ver>) and falling back to the environment variable the Dockerfile sets.
if(NOT DEFINED L4T_MAJ_VER AND DEFINED ENV{L4T_MAJ_VER})
    set(L4T_MAJ_VER "$ENV{L4T_MAJ_VER}")
endif()
# NVSIPL_API_MAJOR_VERSION 1 = L4T R36–R38; 2 = L4T R39+ (new API types + NV_EMBEDDED_L4T)
if(DEFINED L4T_MAJ_VER AND NOT "${L4T_MAJ_VER}" STREQUAL "invalid")
    if("${L4T_MAJ_VER}" GREATER_EQUAL 39)
        set(NVSIPL_API_MAJOR_VERSION 2)
    else()
        set(NVSIPL_API_MAJOR_VERSION 1)
    endif()
    message(STATUS "NvSIPL API version: ${NVSIPL_API_MAJOR_VERSION} (L4T R${L4T_MAJ_VER})")
endif()

# SIPL user-space libs live in the L4T BSP path on Jetson/IGX devices.
# On a SIPL-capable build host they are present; on non-SIPL builder nodes they may not be —
# --allow-shlib-undefined below handles that case.
find_library(NVSIPL_LIB nvsipl
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)
find_library(NVSIPL_QUERY_LIB nvsipl_query
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)

if(NOT NVSIPL_LIB OR NOT NVSIPL_QUERY_LIB)
    message(WARNING
        "libnvsipl.so / libnvsipl_query.so not found — "
        "building with --allow-shlib-undefined. "
        "Mount the L4T BSP at /usr/lib/aarch64-linux-gnu/nvidia for runtime.")
endif()

if(DEFINED NVSIPL_API_MAJOR_VERSION)
    # PUBLIC so that consumers including sipl_capture_service.hpp (which transitively
    # pulls in sipl_compat.hpp) compile sipl_compat.hpp's #if SIPL_V2 correctly.
    target_compile_definitions(sipl_capture_op PUBLIC
        NVSIPL_API_MAJOR_VERSION=${NVSIPL_API_MAJOR_VERSION})
    if(NVSIPL_API_MAJOR_VERSION GREATER_EQUAL 2)
        # NV_EMBEDDED_L4T exposes RAW10TP/RAW12TP format constants in NvSIPLCapStructs.h
        target_compile_definitions(sipl_capture_op PUBLIC NV_EMBEDDED_L4T)
    endif()
endif()
# PUBLIC: sipl_capture_service.hpp is a public header that directly includes SIPL/NvSci
# headers, so every consumer needs these paths on its include search list.
target_include_directories(sipl_capture_op PUBLIC
    ${NVSIPL_INCLUDE_DIR}
    ${NVSIPL_INCLUDE_DIR}/nvsci
    ${NVSIPL_INCLUDE_DIR}/nvmedia)

# NvSIPLCameraQuery.hpp moved to query/include/ in R39+ (SIPL API v2)
if(NVSIPL_API_MAJOR_VERSION GREATER_EQUAL 2)
    target_include_directories(sipl_capture_op PUBLIC
        ${NVSIPL_INCLUDE_DIR}/query/include)
endif()

find_library(NVSCIBUF_LIB nvscibuf
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)
find_library(NVSCISYNC_LIB nvscisync
    HINTS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)

if(NOT NVSCIBUF_LIB OR NOT NVSCISYNC_LIB)
    message(WARNING
        "libnvscibuf.so / libnvscisync.so not found — "
        "building with --allow-shlib-undefined.")
endif()

# --allow-shlib-undefined: libnvsipl.so / libnvsipl_query.so have a transitive
# dependency on L4T kernel driver libs not present in the build container.
# The flag lets the linker proceed; drivers are mounted at runtime via the
# holohub wrapper script. Scope the flag tightly to SIPL + NvSci libs only.
target_link_libraries(sipl_capture_op PRIVATE
    "LINKER:--allow-shlib-undefined"
    $<$<BOOL:${NVSIPL_LIB}>:${NVSIPL_LIB}>
    $<$<BOOL:${NVSIPL_QUERY_LIB}>:${NVSIPL_QUERY_LIB}>
    $<$<BOOL:${NVSCIBUF_LIB}>:${NVSCIBUF_LIB}>
    $<$<BOOL:${NVSCISYNC_LIB}>:${NVSCISYNC_LIB}>
    "LINKER:--no-allow-shlib-undefined")
find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
target_link_libraries(sipl_capture_op PRIVATE CUDA::cudart)

target_link_libraries(sipl_capture_op PUBLIC holoscan::core holoscan::schema holoscan::sensor_io holoscan::sensor_io_schemas)

install(FILES sipl_capture_op.hpp sipl_capture_service.hpp sipl_compat.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sipl_capture_op)

# Install every header published by the SDK helper so downstream consumers can use
# Input<SIPLFrameMetadataT> and binary_schema<SIPLFrameMetadataT>() without a build-tree path.
get_target_property(SIPL_METADATA_GENERATED_HEADERS
    sipl_frame_metadata_generated HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)
install(FILES ${SIPL_METADATA_GENERATED_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sipl_capture_op)

install(TARGETS sipl_capture_op
    EXPORT holoscan_camera_targets)

# Tests live with the operator they cover, so an operator added later brings its own without
# editing a shared test directory. Gated here rather than in tests/CMakeLists.txt so that
# find_package(GTest) is not required by a build that only wants the operator.
if(HOLOSCAN_CAMERA_BUILD_TESTING)
    add_subdirectory(tests)
endif()
