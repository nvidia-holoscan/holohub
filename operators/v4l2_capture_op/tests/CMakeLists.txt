# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Guarded because an out-of-tree build inside the Holoscan SDK tree already defines the target, and
# find_package would fail to locate a config for it.
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(test_v4l2_capture_op test_v4l2_capture_op.cpp)

# The operator's own include directory arrives through the linked target, which exports
# ${PROJECT_SOURCE_DIR}/operators as a build-interface include directory, so the test does not
# restate it. holoscan::sensor_io is named directly rather than relied on transitively because the
# restart-contract test uses StageGuards itself.
target_link_libraries(test_v4l2_capture_op PRIVATE
    holoscan::v4l2_capture_op
    holoscan::sensor_io
    holoscan::core
    GTest::gtest_main)

include(GoogleTest)
# The SDK's out-of-source builders compile against CUDA but do not expose a driver or device.
# Discover at CTest time so linking this target never executes it during the build.
gtest_discover_tests(test_v4l2_capture_op
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "unit")
