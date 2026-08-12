# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET v4l2_depth)
    add_test(NAME v4l2_depth_help COMMAND v4l2_depth --help)
    set_tests_properties(v4l2_depth_help
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Live V4L2 YUYV capture"
            TIMEOUT 10
            LABELS "cpp;cli"
    )

    add_test(
        NAME v4l2_depth_loopback_feeder_help
        COMMAND
            "${CMAKE_SOURCE_DIR}/applications/v4l2_depth/scripts/feed_v4l2_loopback.py"
            --help
    )
    set_tests_properties(v4l2_depth_loopback_feeder_help
        PROPERTIES
            PASS_REGULAR_EXPRESSION "1280x720 progressive YUYV"
            TIMEOUT 10
            LABELS "shell;cli"
    )

    add_test(
        NAME v4l2_depth_loopback_runner_help
        COMMAND
            "${CMAKE_SOURCE_DIR}/applications/v4l2_depth/scripts/run_v4l2_depth_loopback.py"
            --help
    )
    set_tests_properties(v4l2_depth_loopback_runner_help
        PROPERTIES
            PASS_REGULAR_EXPRESSION "1280x720 at 25 FPS, without padding"
            TIMEOUT 10
            LABELS "python;cli"
    )

    # Compilation exercises the full typed graph and explicit CUDA placement
    # without invoking lifecycle start(), so no camera, model, or X server is
    # required for this test.
    add_test(NAME v4l2_depth_graph_validation COMMAND v4l2_depth --validate)
    set_tests_properties(v4l2_depth_graph_validation
        PROPERTIES
            PASS_REGULAR_EXPRESSION "v4l2_depth graph validation complete"
            TIMEOUT 20
            LABELS "cpp;graph"
    )

    # Exercise metadata/default-mode resolution separately from the binary so
    # a regression that routes plain `holohub run v4l2_depth` to the live
    # camera mode cannot hide behind the direct --validate test above.
    add_test(
        NAME v4l2_depth_default_mode
        COMMAND
            "${CMAKE_COMMAND}" -E env HOLOSCAN_CLI_BUILD_LOCAL=1
            "${CMAKE_SOURCE_DIR}/holohub" run v4l2_depth --dryrun
    )
    set_tests_properties(v4l2_depth_default_mode
        PROPERTIES
            PASS_REGULAR_EXPRESSION "--validate"
            TIMEOUT 20
            LABELS "cli;metadata"
    )

    add_test(
        NAME v4l2_depth_validate_container_target
        COMMAND
            "${CMAKE_COMMAND}" -E env --unset=HOLOSCAN_CLI_BUILD_LOCAL
            "${CMAKE_SOURCE_DIR}/holohub" build v4l2_depth --dryrun
    )
    set_tests_properties(v4l2_depth_validate_container_target
        PROPERTIES
            PASS_REGULAR_EXPRESSION "v4l2-depth-validate"
            TIMEOUT 20
            LABELS "cli;metadata"
    )

    add_test(
        NAME v4l2_depth_live_container_target
        COMMAND
            "${CMAKE_COMMAND}" -E env --unset=HOLOSCAN_CLI_BUILD_LOCAL
            "${CMAKE_SOURCE_DIR}/holohub" build v4l2_depth live --dryrun
    )
    set_tests_properties(v4l2_depth_live_container_target
        PROPERTIES
            PASS_REGULAR_EXPRESSION "v4l2-depth-live"
            TIMEOUT 20
            LABELS "cli;metadata"
    )

    add_test(
        NAME v4l2_depth_loopback_container_target
        COMMAND
            "${CMAKE_COMMAND}" -E env --unset=HOLOSCAN_CLI_BUILD_LOCAL
            "${CMAKE_SOURCE_DIR}/holohub" build v4l2_depth loopback --dryrun
    )
    set_tests_properties(v4l2_depth_loopback_container_target
        PROPERTIES
            PASS_REGULAR_EXPRESSION "v4l2-depth-loopback"
            TIMEOUT 20
            LABELS "cli;metadata"
    )

    add_test(
        NAME v4l2_depth_loopback_mode_command
        COMMAND
            "${CMAKE_COMMAND}" -E env HOLOSCAN_CLI_BUILD_LOCAL=1
            "${CMAKE_SOURCE_DIR}/holohub" run v4l2_depth loopback --dryrun
    )
    set_tests_properties(v4l2_depth_loopback_mode_command
        PROPERTIES
            PASS_REGULAR_EXPRESSION "run_v4l2_depth_loopback.py"
            TIMEOUT 20
            LABELS "cli;metadata"
    )
endif()
