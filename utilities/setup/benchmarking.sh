#!/usr/bin/env bash

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

set -e

# =========================================================
# Install standard Holoscan flow benchmarking dependencies.
# =========================================================

# Install APT dependencies
apt update
apt install --no-install-recommends -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    libgtk-3-dev \
    libcanberra-gtk-module \
    graphviz

# Install Python dependencies
python3 -m pip install \
    meson \
    numpy \
    matplotlib \
    nvitop \
    pydot \
    PyGObject==3.50.0 \
    xdot
if ! grep -q 'VERSION_ID="22.04"' /etc/os-release; then
    python3 -m pip install setuptools
fi
