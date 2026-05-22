# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS holohub-cli

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# Set up GStreamer-specific dependencies
#
# --------------------------------------------------------------------------
RUN holoscan version \
    && test -x /tmp/scripts/holohub

# Install GStreamer development packages and pkg-config
COPY operators/gstreamer/install_deps.sh /tmp/install_deps.sh
RUN /tmp/install_deps.sh

# Build tooling for C++ development and Python binding compilation
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        clang-format \
        pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tooling: pytest for tests, build + scikit-build-core for wheel packaging
RUN python3 -m pip install --no-cache-dir \
        pytest \
        pytest-timeout \
        build \
        scikit-build-core

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# Set default Holohub data directory
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
