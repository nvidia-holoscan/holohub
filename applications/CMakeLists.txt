# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each application is gated on its own APP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_application() call per application; declare operator
# dependencies via DEPENDS OPERATORS so they auto-enable when the app is built.
add_holohub_application(holoscan_camera_v4l2
    DEPENDS OPERATORS v4l2_capture_op
)
add_holohub_application(sipl_frame_saver
    DEPENDS OPERATORS sipl_capture_op
)
add_holohub_application(sipl_stereo_monitor
    DEPENDS OPERATORS sipl_capture_op
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_holohub_application(holoscan_camera_argus
        DEPENDS OPERATORS argus_capture_op
    )
elseif(APP_holoscan_camera_argus)
    message(FATAL_ERROR "holoscan_camera_argus requires an ARM64 target (aarch64 or arm64)")
else()
    message(STATUS "Skipping holoscan_camera_argus: requires an ARM64 target")
endif()
