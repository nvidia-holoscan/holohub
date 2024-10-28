# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG GPU_TYPE=dgpu
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.07-0.6.1.0
ARG CACHEBUST=1
ARG BUILDARCH
RUN echo "Using build architecture ${BUILDARCH}"
RUN echo "Using architecture $TARGETPLATFORM"


RUN apt update && apt install -y python3-pyelftools ninja-build meson
RUN pip3 install scipy loguru attrs
WORKDIR /opt

RUN    eval `dpkg-architecture` \
        && wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.7.0/host/doca-host_2.7.0-204000-24.04-ubuntu2204_${BUILDARCH}.deb -O doca-host.deb \
        && ls -lh && apt install ./doca-host.deb \
        && apt update \
        && apt install -y doca-all \
        && apt install -y doca-gpu doca-gpu-dev

# ==============================
# DOCA Target (inherits from base)
# This stage is only built when --target doca is specified. It contains any DOCA-specific configurations.
# ==============================
FROM base AS doca
# DOCA-specific installation or steps (if needed)

# ==============================
# DPDK Target (inherits from base)
# This stage is only built when --target dpdk is specified. It contains any DPDK-specific configurations.
# ==============================
FROM base AS dpdk
# DPDK-specific installation or steps (if needed)

# ==============================
# Rivermax Target (inherits from base)
# This stage is only built when --target rivermax is specified. It installs and configures Rivermax SDK.
# ==============================
FROM base AS rivermax

# Define Rivermax-specific build arguments and environment variables
ARG RIVERMAX_VERSION=1.60.6
ARG RIVERMAX_SDK_ZIP_PATH=./rivermax_ubuntu2204_${RIVERMAX_VERSION}.tar.gz
ARG MAXPROC=8


# RMAX_TEGRA controls whether the build targets NVIDIA's Tegra platform (default is OFF).
# Affects Rivermax sample apps (via build args) and HoloHub (via environment variable).
# Set to ON/TRUE to target Tegra devices like Jetson.
ARG RMAX_TEGRA=OFF
ENV RMAX_TEGRA=${RMAX_TEGRA}

# Install additional dependencies required for Rivermax
RUN apt install -y iproute2 libcap-dev gdb ethtool iputils-ping net-tools \
    libfmt-dev libnl-3-dev libnl-genl-3-dev libcap-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libglew-dev

# Copy and extract the Rivermax SDK
COPY ${RIVERMAX_SDK_ZIP_PATH} /tmp/rivermax_sdk.tar.gz
RUN if [ -f "/tmp/rivermax_sdk.tar.gz" ]; then \
      echo "Extracting Rivermax SDK..." && \
      tar -xzf /tmp/rivermax_sdk.tar.gz && \
      mv /opt/${RIVERMAX_VERSION} /opt/rivermax_sdk && \
      rm -v /tmp/rivermax_sdk.tar.gz; \
    else \
      echo "Error: Rivermax SDK tar.gz not found in /tmp"; exit 1; \
    fi

WORKDIR /opt/rivermax_sdk

# Find and install the Rivermax .deb package based on the build architecture
RUN DEB_FILE=$(find . -name "rivermax_${RIVERMAX_VERSION}_${BUILDARCH}.deb" -type f) && \
    if [ -f "$DEB_FILE" ]; then \
        echo "Installing Rivermax core from $DEB_FILE..." && \
        dpkg -i "$DEB_FILE"; \
    else \
        echo "Error: Rivermax ${BUILDARCH}.deb package not found"; exit 1; \
    fi

# Build Rivermax test and sample applications
RUN cd apps && \
    cmake -B build -DRMAX_CUDA=ON -DRMAX_TEGRA=${RMAX_TEGRA} -DRMAX_BUILD_VIEWER=ON && \
    cmake --build build -j $(nproc)

# Temporarily add missing definitions for rmax_apps_lib build until added to the SDK
RUN echo '\
    target_compile_definitions(rmax-apps-util-reduced PUBLIC \
            \$<\$<BOOL:\${RMAX_CUDA}>:CUDA_ENABLED> \
            \$<\$<BOOL:\${RMAX_TEGRA}>:TEGRA_ENABLED> \
    )' >> rmax_apps_lib/util/CMakeLists.txt

# Build rmax_apps_lib sample applications
RUN cd rmax_apps_lib && \
    cmake -B build -DRMAX_CUDA=ON -DRMAX_TEGRA=${RMAX_TEGRA} -DRMAX_BUILD_VIEWER=ON && \
    cmake --build build -j $(nproc)

# Temporarily exclude apps from rmax_apps_lib build until added to the SDK
RUN sed -i 's/\"apps\"//' rmax_apps_lib/CMakeLists.txt

# ==============================
# Default stage: Base
# If no target is specified, the base stage will be built by default.
# ==============================
FROM base
