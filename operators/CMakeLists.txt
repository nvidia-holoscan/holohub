# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each operator is gated on its own OP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_operator() call per operator.
add_holohub_operator(v4l2_capture_op)
add_holohub_operator(sipl_capture_op)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_holohub_operator(argus_capture_op)
elseif(OP_argus_capture_op)
    message(FATAL_ERROR "argus_capture_op requires an ARM64 target (aarch64 or arm64)")
else()
    message(STATUS "Skipping argus_capture_op: requires an ARM64 target")
endif()
