# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# 0.2.0 because the operator's published payload changed from holoscan::Tensor to
# holoscan::schema::ImageT, its tensor rank from three to two, and its default placement from CUDA
# device to host. This version reaches the Debian package through PROJECT_VERSION, so leaving it at
# 0.1.0 would ship that break under the identity of the interface it breaks. Keep it equal to
# metadata.json:module.version and to the operator and application metadata versions.
project(holoscan_camera
    VERSION 0.2.0
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
# refused by name -- and ci/holoscan-sdk.version pins the exact qualified revision.
if(NOT TARGET holoscan::core)
    find_package(holoscan 5.0.0 REQUIRED CONFIG COMPONENTS sensor_io)
endif()


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
