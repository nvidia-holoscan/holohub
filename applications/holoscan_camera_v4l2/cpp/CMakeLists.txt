# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(holoscan_camera_v4l2 holoscan_camera_v4l2.cpp)

target_compile_features(holoscan_camera_v4l2 PRIVATE cxx_std_20)
# holoscan::sensor_io is named directly here, not merely inherited from the operator: the survey
# sink declares an Input<schema::ImageT>, and the registration that makes ImageT a legal port
# payload lives in this component rather than in Core.
target_link_libraries(holoscan_camera_v4l2 PRIVATE
    holoscan::v4l2_capture_op
    holoscan::sensor_io
    holoscan::core)

target_include_directories(holoscan_camera_v4l2 PRIVATE
    ${PROJECT_SOURCE_DIR}/operators)

install(TARGETS holoscan_camera_v4l2
    RUNTIME DESTINATION bin)

# The unit tests cover the operator's own validation of memory_kind, but nothing there compiles a
# plan, so the part of a placement that can only fail at compile time is otherwise unchecked: that
# the source's declared placement and the sink's expectation agree, and that the CUDA-device
# binding the application supplies is required for kCudaDevice and refused for the host kinds.
# --validate compiles the graph and opens no device, so it establishes exactly that on a runner
# with no camera attached.
#
# Registered per placement rather than as one test so a failure names the placement that broke, and
# matched against the reported label rather than trusting the exit status alone, since a placement
# that silently fell back to the default would still compile and exit zero.
if(HOLOSCAN_CAMERA_BUILD_TESTING)
    foreach(placement IN ITEMS host pinned device)
        # kCudaDevice reports the bound ordinal alongside the kind, the host kinds bind nothing.
        set(expected_placement "${placement}")
        set(placement_labels "plan")
        if(placement STREQUAL "device")
            set(expected_placement "device:0")
            # A device-resident pool needs a real GPU, unlike the host placements, so a runner
            # without one can exclude this with `ctest -LE gpu` instead of failing.
            list(APPEND placement_labels "gpu")
        endif()

        add_test(NAME holoscan_camera_v4l2_validate_${placement}
            COMMAND holoscan_camera_v4l2 --validate --memory-kind ${placement})
        set_tests_properties(holoscan_camera_v4l2_validate_${placement} PROPERTIES
            LABELS "${placement_labels}"
            PASS_REGULAR_EXPRESSION
                "graph validation complete \\(memory=${expected_placement}\\)")
    endforeach()
endif()
