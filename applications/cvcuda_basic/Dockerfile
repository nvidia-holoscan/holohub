# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

############################################################
# CV-CUDA Downloader
############################################################
FROM ${BASE_IMAGE} as cvcuda-install

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/nvidia

# Detect architecture and install appropriate CV-CUDA wheel
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        # install Python package for x86_64
        pip install cvcuda-cu12==0.15.0 && \
        # Download and install C++ library for x86_64
        curl -LO https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-lib-0.15.0-cuda12-x86_64-linux.deb && curl -LO https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-dev-0.15.0-cuda12-x86_64-linux.deb && \
        dpkg -i cvcuda-lib-0.15.0-cuda12-x86_64-linux.deb && \
        dpkg -i cvcuda-dev-0.15.0-cuda12-x86_64-linux.deb && \
        rm cvcuda-lib-0.15.0-cuda12-x86_64-linux.deb && \
        rm cvcuda-dev-0.15.0-cuda12-x86_64-linux.deb; \
    else \
        # install Python package for aarch64
        pip install cvcuda-cu12==0.15.0 && \
        # Download and install C++ library for aarch64
        curl -LO https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-lib-0.15.0-cuda12-aarch64-linux.deb && \
        curl -LO https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-dev-0.15.0-cuda12-aarch64-linux.deb && \
        dpkg -i cvcuda-lib-0.15.0-cuda12-aarch64-linux.deb && \
        dpkg -i cvcuda-dev-0.15.0-cuda12-aarch64-linux.deb && \
        rm cvcuda-lib-0.15.0-cuda12-aarch64-linux.deb && \
        rm cvcuda-dev-0.15.0-cuda12-aarch64-linux.deb; \
    fi

############################################################
# Base (final)
############################################################
FROM cvcuda-install

ARG DEBIAN_FRONTEND=noninteractive

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
