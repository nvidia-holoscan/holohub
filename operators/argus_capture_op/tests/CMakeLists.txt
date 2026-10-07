# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()
add_executable(test_argus_capture_op argus_capture_op_test.cpp)
# The production operator objects, with only the Jetson backend replaced by a deterministic fake.
target_link_libraries(test_argus_capture_op PRIVATE argus_capture_impl GTest::gtest_main)
if(HOLOSCAN_CAMERA_RUN_ARGUS_TESTS)
    include(GoogleTest)
    gtest_discover_tests(test_argus_capture_op DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "unit;argus" TIMEOUT 15)
endif()
