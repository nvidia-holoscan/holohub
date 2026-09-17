# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-camera ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_camera is ON.
include(holohub_configure_deb)

holohub_configure_deb(
    NAME        "holoscan-camera"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "NVIDIA"
    CONTACT     "jjomier@nvidia.com"
    DEPENDS     "holoscan (>= 5.0.0)"
    EXPORT_NAME "holoscan_camera_targets"
)
