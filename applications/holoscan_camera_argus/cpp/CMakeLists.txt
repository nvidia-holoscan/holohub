# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_executable(holoscan_camera_argus holoscan_camera_argus.cpp)
target_compile_features(holoscan_camera_argus PRIVATE cxx_std_20)
target_link_libraries(holoscan_camera_argus PRIVATE holoscan::argus_capture_op)
include(GNUInstallDirs)
install(TARGETS holoscan_camera_argus RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(HOLOSCAN_CAMERA_BUILD_TESTING AND HOLOSCAN_CAMERA_RUN_ARGUS_TESTS)
    foreach(format IN ITEMS nv12 i420)
        add_test(NAME argus_validate_${format}
            COMMAND holoscan_camera_argus --validate --memory host --format ${format})
        set_tests_properties(argus_validate_${format} PROPERTIES LABELS "plan;argus")
    endforeach()
    add_test(NAME argus_validate_device
        COMMAND holoscan_camera_argus --validate --memory device)
    set_tests_properties(argus_validate_device PROPERTIES LABELS "plan;gpu;argus")
endif()
