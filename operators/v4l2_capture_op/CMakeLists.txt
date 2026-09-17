# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(v4l2_capture_op v4l2_capture_op.cpp)
add_library(holoscan::v4l2_capture_op ALIAS v4l2_capture_op)

target_compile_features(v4l2_capture_op PUBLIC cxx_std_20)
target_include_directories(v4l2_capture_op PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/operators>
    $<INSTALL_INTERFACE:include>)

# holoscan::sensor_io supplies the acquisition discipline the operator's header exposes as members
# (StageGuards, ClockDiscipline, SequenceTracker) and the ImageT payload on the frame port, so it is
# a PUBLIC dependency rather than an implementation detail. It resolves from the find_package(holoscan)
# in the top-level CMakeLists, which names sensor_io as a component so that an SDK build lacking it is
# reported there rather than as a missing target on the line below.
target_link_libraries(v4l2_capture_op PUBLIC
    holoscan::core
    holoscan::sensor_io)

# v4l2_device.hpp is deliberately absent: it is an implementation detail that would otherwise pull
# linux/videodev2.h, sys/mman.h, and poll.h into the include path of every consumer of the operator.
install(FILES v4l2_capture_op.hpp DESTINATION include/v4l2_capture_op)

# Add the operator to an export set. The package layer installs the set and
# generates the CMake config — see holohub_configure_deb(... EXPORT_NAME ...) in
# pkg/holoscan-camera/CMakeLists.txt — so downstream projects
# can find_package(holoscan-camera) and link
# holoscan::v4l2_capture_op.
install(TARGETS v4l2_capture_op
    EXPORT holoscan_camera_targets)

# Tests live with the operator they cover, so an operator added later brings its own without
# editing a shared test directory. Gated here rather than in tests/CMakeLists.txt so that
# find_package(GTest) is not required by a build that only wants the operator.
if(HOLOSCAN_CAMERA_BUILD_TESTING)
    add_subdirectory(tests)
endif()
