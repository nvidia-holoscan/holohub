# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# 0.5.0 adds an installable module surface for SDK nesting without changing the
# operator or application APIs.
# This version reaches the Debian package through PROJECT_VERSION, so it tracks
# the module-level release, not any single operator or
# application -- those keep their own versions in their own metadata.json, independent of this
# one. Keep this equal to metadata.json:module.version at the repository root.
project(holoscan_camera
    VERSION 0.5.0
    DESCRIPTION "Concrete camera implementations for Holoscan SDK 5.0 EA"
    LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# holoscan::sensor_io is named as a component even though the full package loads it by default,
# because that default is silent when it fails. Sensor I/O ships from its own export file and the
# config treats a missing one as a non-error, so an SDK build without it satisfies a plain REQUIRED
# find_package and then fails at the first target_link_libraries with a missing-target message that
# names no cause. Requesting the component moves that failure here, where the reason is reported.
#
# Guarded because an out-of-tree build inside the Holoscan SDK tree already defines these targets,
# and there is no installed config for find_package to locate in that arrangement.
#
# The version floor stays 5.0.0 even though this module needs Sensor I/O features newer than the
# 5.0.0 release, because there is no higher number to ask for: the SDK's VERSION file reads 5.0.0
# at every revision that carries them, so v5.0.0.2 is a git tag and not a package version, and
# requesting it here would fail configuration against the very SDK that satisfies the dependency.
# The component request above is what actually enforces the floor -- an SDK without sensor_io is
# refused by name. See RELEASE_NOTES.md for the SDK source baseline.
if(NOT TARGET holoscan::core)
    find_package(holoscan 5.0.0 REQUIRED CONFIG COMPONENTS sensor_io)
endif()

# ---------------------------------------------------------------------------
# Install layout
# ---------------------------------------------------------------------------
# Every install rule carries the component so a parent build can
# `cmake --install --component holoscan-camera`; destinations come from
# GNUInstallDirs, which is also what the SDK's own install rules use.
include(GNUInstallDirs)
set(HOLOSCAN_CAMERA_INSTALL_COMPONENT holoscan-camera)
option(HOLOSCAN_CAMERA_BUILD_DEB
    "Configure the holoscan-camera Debian package (CPack) even when this module is built nested in another project"
    OFF)
# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
# BUILD_ALL toggles every operator/application/package on by default. It
# matches the HoloHub-monorepo convention: each subproject is a CMake option
# (e.g. APP_<name>, OP_<name>, PKG_<name>) defined via add_holohub_application
# / add_holohub_operator / add_holohub_package, defaulting to ${BUILD_ALL}.
# To build a single subproject, configure with -DBUILD_ALL=OFF -DAPP_<name>=ON.
#
# Default to ON only when this is the top-level CMake project (standalone
# build). When nested under another project (e.g. consumed as a HoloHub
# external dependency via add_subdirectory), defer to whatever the parent
# build has decided — otherwise this module's BUILD_ALL=ON would leak into
# the parent's variable scope and force-enable unrelated subprojects there.

if(NOT DEFINED BUILD_ALL)
    if(PROJECT_IS_TOP_LEVEL)
        set(BUILD_ALL ON CACHE BOOL "Build all operators, applications, and packages by default")
    else()
        set(BUILD_ALL OFF)
    endif()
endif()

# SIPL API v2 is supported only on L4T R39 and later.  The container exposes
# its L4T major version through L4T_MAJ_VER, so an R38 image can build the
# V4L2 application without configuring SIPL targets whose headers are absent.
# These cache entries must be set before applications/ and operators/ define
# their options below.
if(NOT DEFINED L4T_MAJ_VER AND DEFINED ENV{L4T_MAJ_VER})
    set(L4T_MAJ_VER "$ENV{L4T_MAJ_VER}")
endif()
if(NOT DEFINED L4T_MAJ_VER AND EXISTS "/etc/nv_tegra_release")
    file(STRINGS "/etc/nv_tegra_release" _l4t_release LIMIT_COUNT 1)
    if(_l4t_release MATCHES "^# R([0-9]+)")
        set(L4T_MAJ_VER "${CMAKE_MATCH_1}")
    endif()
endif()
if(DEFINED L4T_MAJ_VER AND L4T_MAJ_VER MATCHES "^[0-9]+$" AND L4T_MAJ_VER LESS 39)
    message(STATUS "L4T R${L4T_MAJ_VER}: SIPL disabled; V4L2 available")
    message(WARNING "Argus capture is not verified on IGX R38; its targets require development headers and link libraries")
    set(APP_sipl_frame_saver OFF CACHE BOOL "SIPL requires L4T R39+" FORCE)
    set(APP_sipl_stereo_monitor OFF CACHE BOOL "SIPL requires L4T R39+" FORCE)
    set(OP_sipl_capture_op OFF CACHE BOOL "SIPL requires L4T R39+" FORCE)
endif()

# Module tests live behind a module-namespaced flag rather than the standard
# CMake BUILD_TESTING variable. This lets a parent project (HoloHub, another
# module) keep its own BUILD_TESTING setting independent of whether *this*
# module's tests are built. Defaults ON when standalone, OFF when nested.
if(NOT DEFINED HOLOSCAN_CAMERA_BUILD_TESTING)
    if(PROJECT_IS_TOP_LEVEL AND BUILD_ALL)
        set(HOLOSCAN_CAMERA_BUILD_TESTING ON
            CACHE BOOL "Build the holoscan_camera module's CTest suites")
    else()
        set(HOLOSCAN_CAMERA_BUILD_TESTING OFF
            CACHE BOOL "Build the holoscan_camera module's CTest suites")
    endif()
endif()

# Keep Argus build coverage on ARM64 CI runners without a Jetson runtime. This controls
# test registration only; HOLOSCAN_CAMERA_BUILD_TESTING still builds the unit-test executable.
option(HOLOSCAN_CAMERA_RUN_ARGUS_TESTS
    "Register Argus tests for execution with a compatible Jetson runtime" ON)

# ---------------------------------------------------------------------------
# CMake helpers
# ---------------------------------------------------------------------------
# HoloHubConfigHelpers.cmake provides add_holohub_application/operator/package
# which gate each subproject on its own option (APP_/OP_/PKG_<name>). The
# helpers ship with the HoloHub CLI and are copied into this module's cmake/
# directory by the cookiecutter post-gen hook so the project also builds
# without a HoloHub clone.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HoloHubConfigHelpers)

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
# pkg/ must come *before* applications/ and operators/: add_holohub_package()
# inside pkg/ uses set(... CACHE BOOL "" FORCE) to flip OP_<x>=ON / APP_<x>=ON
# when its PKG_<name> option is ON, and the operator/application macros that
# read those values run their `option(OP_<x> ... ${BUILD_ALL})` calls when
# their subdirectories are processed — those bind the cache value the FIRST
# time option() is encountered, so the FORCE has to happen before then.
# applications/ then runs before operators/ so app-declared operator
# dependencies are visible before add_holohub_operator() evaluates OP_<name>.

# Before the subdirectories, not after: tests now live beside the operator they cover, so they are
# declared while operators/ is processed. CTest only records a test whose add_test() runs after
# enable_testing(), and a test declared earlier is silently absent from the test list rather than an
# error, so the order here is what makes those tests exist at all.
if(HOLOSCAN_CAMERA_BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(pkg)
add_subdirectory(applications)
add_subdirectory(operators)

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
# Debian packaging lives under pkg/. pkg/CMakeLists.txt calls
# add_holohub_package(holoscan-camera OPERATORS v4l2_capture_op
# APPLICATIONS holoscan_camera_v4l2), which defines PKG_holoscan_camera,
# FORCEs the OP_/APP_ deps to ON when the package is enabled, and enters
# pkg/holoscan-camera/CMakeLists.txt for the holohub_configure_deb() call.
