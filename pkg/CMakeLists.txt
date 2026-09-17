# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# add_holohub_package() handles the full cascade for both standalone
# packaging (`./holoscan_camera package holoscan-camera --pkg-generator DEB`, which
# sets -DPKG_holoscan_camera=ON via the HoloHub CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency:
#
#   1. Defines option(PKG_holoscan_camera ... ${BUILD_ALL}) — ON
#      when BUILD_ALL=ON (default standalone) or when the CLI passes the
#      option explicitly.
#   2. add_subdirectory(holoscan-camera) — enters
#      pkg/holoscan-camera/ where holohub_configure_deb() emits
#      CPackConfig-holoscan-camera.cmake.
#   3. set("OP_<x>" ON CACHE BOOL ... FORCE) and the equivalent for
#      APPLICATIONS — cascades to the operator and application subprojects
#      so the .deb gets actual content. pkg/ is add_subdirectory()'d
#      *before* operators/ and applications/, so those see the forced ON
#      value when their own helpers run option().
add_holohub_package(holoscan-camera
                    OPERATORS    v4l2_capture_op
                    APPLICATIONS holoscan_camera_v4l2)
