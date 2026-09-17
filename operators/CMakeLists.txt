# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each operator is gated on its own OP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_operator() call per operator.
add_holohub_operator(v4l2_capture_op)
