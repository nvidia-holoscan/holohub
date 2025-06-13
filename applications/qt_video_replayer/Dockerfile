# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE
ARG GPU_TYPE

# Pin to Holoscan 3.2 since application requires CUDA Driver 550+ for Holoscan 3.3+
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu as base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Install Qt6
# (when on Ubuntu 20.04 use qt6-backports PPA)
RUN . /etc/lsb-release \
    && if [ "$DISTRIB_RELEASE" = "20.04" ]; then \
      apt update \
      && apt install --no-install-recommends -y \
        software-properties-common="0.99.*" \
      && add-apt-repository ppa:okirby/qt6-backports -y; \
    fi \
    && apt update \
    && apt install --no-install-recommends -y \
        qt6-base-dev \
        qt6-declarative-dev \
        qml6-module-qtquick-controls \
        qml6-module-qtquick-layouts \
        qml6-module-qtquick-templates \
        qml6-module-qtquick-window \
        qml6-module-qtqml-workerscript \
        libqt6opengl6-dev \
        libxkbcommon-dev \
        libgl-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
#
# Holohub run setup
#

RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release
