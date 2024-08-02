#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

find_package(holoscan 2.1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(pva_video_filter
  main.cpp
)

target_link_libraries(pva_video_filter
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::video_stream_recorder
  holoscan::ops::holoviz
)

add_library(pva_unsharp_mask STATIC IMPORTED)

# Define the location in the build directory where libpva_unsharp_mask.a will be used
set(PVA_UNSHARP_MASK_LIB_DEST "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask/libpva_unsharp_mask.a")
# Define the destination path in the build directory for cupva_allowlist_pva_unsharp_mask
set(CUPVA_ALLOWLIST_DEST "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask/cupva_allowlist_pva_unsharp_mask")

# Define the URL for downloading libpva_unsharp_mask.a if it's not found in the source directory
set(PVA_UNSHARP_MASK_URL "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/libpva_unsharp_mask.a")
# Define the URL for downloading cupva_allowlist_pva_unsharp_mask
set(CUPVA_ALLOWLIST_URL "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/cupva_allowlist_pva_unsharp_mask")

# Define a custom command for preparing libpva_unsharp_mask.a and cupva_allowlist_pva_unsharp_mask
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask/prepared.stamp"
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Preparing PVA dependencies..."
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask"
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Directory ensured at ${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask"
  COMMAND ${CMAKE_COMMAND} -D PVA_UNSHARP_MASK_LIB_DEST="${PVA_UNSHARP_MASK_LIB_DEST}" -D PVA_UNSHARP_MASK_URL="${PVA_UNSHARP_MASK_URL}" -D CUPVA_ALLOWLIST_URL="${CUPVA_ALLOWLIST_URL}" -D CUPVA_ALLOWLIST_DEST="${CUPVA_ALLOWLIST_DEST}" -P "${CMAKE_CURRENT_LIST_DIR}/PreparePVADependencies.cmake"
  COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask/prepared.stamp"
  COMMENT "Preparing libpva_unsharp_mask.a and cupva_allowlist_pva_unsharp_mask"
)

# Define a custom target that depends on the custom command
add_custom_target(prepare_pva_dependencies ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/pva_unsharp_mask/prepared.stamp"
)

# Ensure the custom target is built before the main target
add_dependencies(pva_video_filter prepare_pva_dependencies)

# Update the IMPORTED_LOCATION to the new path in the build directory
set_target_properties(pva_unsharp_mask PROPERTIES IMPORTED_LOCATION ${PVA_UNSHARP_MASK_LIB_DEST})

# add according to your CUPVA version here
find_library(CUPVAHOST_LIB libcupva_host.so.2.5 PATHS /opt/nvidia/cupva-2.5/lib/aarch64-linux-gnu/ REQUIRED)

target_link_libraries(pva_video_filter
  PUBLIC
  pva_unsharp_mask
  ${CUPVAHOST_LIB}
)

# Copy the config to the binary directory
add_custom_target(pva_video_filter_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/main.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "main.yaml"
  BYPRODUCTS "main.yaml"
)
add_dependencies(pva_video_filter pva_video_filter_deps)
