# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Guarded because an out-of-tree build inside the Holoscan SDK tree already defines the target, and
# find_package would fail to locate a config for it.
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

# --allow-shlib-undefined: libsipl_capture_op.so has transitive deps on L4T kernel libs
#   that are absent in the build container but mounted at runtime.
# --disable-new-dtags: emit DT_RPATH instead of DT_RUNPATH so the L4T path propagates
#   to transitive deps (e.g. libnvsipl.so -> libnvrm_mem.so).
set(_sipl_l4t_libs
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu/tegra)
string(REPLACE ";" ":" _sipl_l4t_libs_colon "${_sipl_l4t_libs}")

# ---------------------------------------------------------------------------
# SIPLCaptureOp -- interface unit tests (SIPL libs must be loadable; no camera needed)
# ---------------------------------------------------------------------------
add_executable(sipl_capture_op_test sipl_capture_op_test.cpp)

target_link_libraries(sipl_capture_op_test PRIVATE
    holoscan::sipl_capture_op
    holoscan::core
    GTest::gtest_main)

target_link_options(sipl_capture_op_test PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--disable-new-dtags")
set_target_properties(sipl_capture_op_test PROPERTIES
    BUILD_RPATH "${_sipl_l4t_libs_colon}")

# PRE_TEST defers test discovery to run time so the binary is not executed during the build.
# Required here because libsipl_capture_op.so has transitive dependencies on L4T kernel libs
# that are mounted at runtime but absent in the build container.
include(GoogleTest)
gtest_discover_tests(sipl_capture_op_test
    PROPERTIES LABELS "unit"
    DISCOVERY_MODE PRE_TEST)

# ---------------------------------------------------------------------------
# SIPLCaptureOp -- hardware integration tests (require connected SIPL camera)
#
# These tests skip automatically when SIPL_CAMERA_CONFIG is not set.
# Run with: SIPL_CAMERA_CONFIG=ov2311_raw ctest -L hardware
# ---------------------------------------------------------------------------
add_executable(sipl_capture_op_hardware_test sipl_capture_op_hardware_test.cpp)

target_link_libraries(sipl_capture_op_hardware_test PRIVATE
    holoscan::sipl_capture_op
    holoscan::core
    GTest::gtest_main)

target_link_options(sipl_capture_op_hardware_test PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--disable-new-dtags")
set_target_properties(sipl_capture_op_hardware_test PROPERTIES
    BUILD_RPATH "${_sipl_l4t_libs_colon}")

gtest_discover_tests(sipl_capture_op_hardware_test
    PROPERTIES LABELS "hardware"
    DISCOVERY_MODE PRE_TEST)
