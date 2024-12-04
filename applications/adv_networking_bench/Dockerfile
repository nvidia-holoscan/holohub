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

# Note: duplicate of ../../operators/advanced_network to facilitate
# the use of ./dev_container build_and_run adv_networking_bench
ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS base
ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.07-0.6.1.0
ARG CACHEBUST=1
ARG TARGETARCH
RUN echo "Using architecture ${TARGETARCH}"

RUN apt update && apt install -y python3-pyelftools ninja-build meson libyaml-cpp-dev
RUN pip3 install scipy loguru attrs
WORKDIR /opt

ARG DEBIAN_FRONTEND=noninteractive

RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/2.8.0/ubuntu22.04/x86_64; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/2.8.0/ubuntu22.04/arm64-sbsa; \
    else \
        echo "Unknown architecture: $TARGETARCH"; \
        exit 1; \
    fi \
    && echo "Using DOCA_REPO_LINK=${DOCA_REPO_LINK}" \
    && LOCAL_GPG_KEY_PATH="/usr/share/keyrings/mellanox-archive-keyring.gpg" \
    && curl -fsSL ${DOCA_REPO_LINK}/GPG-KEY-Mellanox.pub | gpg --dearmor | tee ${LOCAL_GPG_KEY_PATH} \
    && echo "deb [signed-by=${LOCAL_GPG_KEY_PATH}] ${DOCA_REPO_LINK} ./" | tee /etc/apt/sources.list.d/mellanox.list \
    && apt update -y \
    && apt install -y --no-install-recommends doca-sdk-aes-gcm doca-sdk-apsh doca-sdk-argp doca-sdk-comch doca-sdk-comm-channel doca-sdk-common doca-sdk-compress doca-sdk-devemu doca-sdk-dma doca-sdk-dpa doca-sdk-dpdk-bridge doca-sdk-erasure-coding doca-sdk-eth doca-sdk-flow doca-sdk-pcc doca-sdk-rdma doca-sdk-sha doca-sdk-telemetry-exporter doca-sdk-urom doca-apsh-config doca-bench doca-caps doca-comm-channel-admin doca-pcc-counters doca-sha-offload-engine doca-socket-relay doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev rdma-core flexio libyara8

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
RUN DEB_FILE=$(find . -name "rivermax_${RIVERMAX_VERSION}_${TARGETARCH}.deb" -type f) && \
    if [ -f "$DEB_FILE" ]; then \
        echo "Installing Rivermax core from $DEB_FILE..." && \
        dpkg -i "$DEB_FILE"; \
    else \
        echo "Error: Rivermax ${TARGETARCH}.deb package not found"; exit 1; \
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
