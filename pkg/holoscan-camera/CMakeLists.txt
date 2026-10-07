# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-camera ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_camera is ON. A standalone
# build uses HoloHub's Debian packaging helper; a nested build can opt into it
# with HOLOSCAN_CAMERA_BUILD_DEB, otherwise it installs the CMake package
# directly into the SDK module layout without including CPack.
if(PROJECT_IS_TOP_LEVEL OR HOLOSCAN_CAMERA_BUILD_DEB)
    include(holohub_configure_deb)

    holohub_configure_deb(
        NAME        "holoscan-camera"
        DESCRIPTION "${PROJECT_DESCRIPTION}"
        VERSION     "${PROJECT_VERSION}"
        VENDOR      "NVIDIA"
        CONTACT     "jjomier@nvidia.com"
        DEPENDS     "holoscan (>= 5.0.0)"
        EXPORT_NAME "holoscan_camera_targets"
    )
else()
    # Nested under a parent project (the Holoscan SDK build): no Debian packaging here. Install
    # the export set and package config in the SDK module layout so the parent's install tree
    # exposes find_package(holoscan-camera) next to the other holoscan-<module> configs.
    include(CMakePackageConfigHelpers)
    set(_holoscan_camera_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan")
    set(_holoscan_camera_package_dir "${CMAKE_CURRENT_BINARY_DIR}/holoscan-camera-package")
    set(ARG_NAME "holoscan-camera")
    set(ARG_EXPORT_NAME "holoscan-camera-targets")
    configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in"
        "${_holoscan_camera_package_dir}/holoscan-camera-config.cmake"
        INSTALL_DESTINATION "${_holoscan_camera_cmake_install_dir}"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)
    write_basic_package_version_file(
        "${_holoscan_camera_package_dir}/holoscan-camera-config-version.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion)
    install(EXPORT holoscan_camera_targets
        FILE holoscan-camera-targets.cmake
        NAMESPACE holoscan::
        DESTINATION "${_holoscan_camera_cmake_install_dir}"
        COMPONENT ${HOLOSCAN_CAMERA_INSTALL_COMPONENT})
    install(FILES
        "${_holoscan_camera_package_dir}/holoscan-camera-config.cmake"
        "${_holoscan_camera_package_dir}/holoscan-camera-config-version.cmake"
        DESTINATION "${_holoscan_camera_cmake_install_dir}"
        COMPONENT ${HOLOSCAN_CAMERA_INSTALL_COMPONENT})
    unset(ARG_NAME)
    unset(ARG_EXPORT_NAME)
    unset(_holoscan_camera_package_dir)
    unset(_holoscan_camera_cmake_install_dir)
endif()
