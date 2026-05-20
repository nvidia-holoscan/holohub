# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(holohub_configure_deb)

# Debian packaging
set(_deb_depends
    "holoscan-cuda-13 (>= 4.2.0)"
    "libgstreamer1.0-0"
    "libgstreamer-plugins-base1.0-0"
    "libgstreamer-plugins-bad1.0-0"
    "gstreamer1.0-plugins-good"
    "gstreamer1.0-plugins-bad"
    "gstreamer1.0-libav"
    "gstreamer1.0-tools"
)
list(JOIN _deb_depends ", " _deb_depends_str)

holohub_configure_deb(
    NAME        holoscan-gstreamer
    DESCRIPTION "GStreamer bridge operators for video recording and streaming in Holoscan applications"
    VERSION     1.0.0
    VENDOR      "NVIDIA Corporation"
    CONTACT     "NVIDIA Corporation"
    DEPENDS     "${_deb_depends_str}"
)

# Wheel packaging: see pyproject.toml
