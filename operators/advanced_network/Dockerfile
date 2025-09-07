# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Define arguments for holoscan installation
ARG HOLOSCAN_DEB_SRC="remote" # or "local"
ARG HOLOSCAN_DEB_REMOTE_VERSION=3.4.0 # only used if HOLOSCAN_DEB_SRC is "remote"
ARG HOLOSCAN_DEB_LOCAL_PATH="./holoscan.deb" # only used if HOLOSCAN_DEB_SRC is "local"

# ============================================================
# base: Base layer
# ============================================================
FROM nvcr.io/nvidia/cuda:12.6.3-base-ubuntu22.04 AS base
SHELL ["/bin/bash", "-eou", "pipefail", "-c"]

# Basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        python3-pip \
        git \
        build-essential \
    && OS_CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME") \
    && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
    && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
        | gpg --dearmor -o "$KW_KEYRING" \
    && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && rm "$KW_KEYRING" \
    && apt-get install --no-install-recommends -y \
        kitware-archive-keyring \
        cmake="3.*" \
        cmake-data="3.*" \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# hsdk-remote: Install holoscan from remote apt repository
# ============================================================
FROM base AS hsdk-remote

ARG HOLOSCAN_DEB_REMOTE_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        holoscan="${HOLOSCAN_DEB_REMOTE_VERSION}*" \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# hsdk-local: Install holoscan from local deb package
# ============================================================
FROM base AS hsdk-local

ARG HOLOSCAN_DEB_LOCAL_PATH
RUN --mount=type=bind,source=${HOLOSCAN_DEB_LOCAL_PATH},target=/tmp/holoscan.deb \
    apt-get update && apt-get install -y --no-install-recommends \
        /tmp/holoscan.deb \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# hsdk: join hsdk-remote and hsdk-local
# ============================================================
FROM hsdk-${HOLOSCAN_DEB_SRC} AS hsdk-installed

# ==============================================================
# common-deps: common dependencies for the advanced network lib
# ==============================================================
FROM hsdk-installed AS common-deps

ARG TARGETARCH
ARG DOCA_VERSION=2.8.0
ARG CACHEBUST=1
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# Configure the DOCA APT repository
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        DOCA_ARCH="x86_64"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        DOCA_ARCH="arm64-sbsa"; \
    else \
        echo "Unknown architecture: $TARGETARCH"; \
        exit 1; \
    fi \
    && DISTRO=$(. /etc/os-release && echo ${ID}${VERSION_ID}) \
    && DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/${DISTRO}/${DOCA_ARCH} \
    && echo "Using DOCA_REPO_LINK=${DOCA_REPO_LINK}" \
    && LOCAL_GPG_KEY_PATH="/usr/share/keyrings/mellanox-archive-keyring.gpg" \
    && curl -fsSL ${DOCA_REPO_LINK}/GPG-KEY-Mellanox.pub | gpg --dearmor | tee ${LOCAL_GPG_KEY_PATH} > /dev/null \
    && echo "deb [signed-by=${LOCAL_GPG_KEY_PATH}] ${DOCA_REPO_LINK} ./" | tee /etc/apt/sources.list.d/mellanox.list

# APT installs
# - cublas, cufft, cusolver, curand dev libs: for matx
# - ninja-build: for cmake build
# - pkgconf: to import dpdk and doca in CMake
# - mlnx-dpdk-dev: for dpdk and gpunetio backends
# - libdoca-sdk-dma-dev: for gpunetio backend (dependency of doca-gpunetio module)
# - libdoca-sdk-gpunetio-dev: for gpunetio backend (doca-gpunetio module)
# - libdoca-sdk-eth-dev: for gpunetio backend (doca-eth module)
# - libdoca-sdk-flow-dev: for gpunetio backend (doca-flow module)
# - mlnx-ofed-kernel-utils: utilities, including ibdev2netdev used by tune_system.py
# - ibverbs-utils: utilities
# - python3-pyelftools: used by some mlnx tools
# - python3-dev: for building python bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcublas-dev-12-6 \
        libcufft-dev-12-6 \
        libcusolver-dev-12-6 \
        libcurand-dev-12-6 \
        ninja-build \
        pkgconf \
        mlnx-dpdk-dev \
        libdoca-sdk-dma-dev \
        libdoca-sdk-gpunetio-dev \
        libdoca-sdk-eth-dev \
        libdoca-sdk-flow-dev \
        mlnx-ofed-kernel-utils \
        ibverbs-utils \
        python3-pyelftools \
        python3-dev \
        libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# PIP installs
# - pytest: test harness
# - pyyaml: to parse yaml configs in tests
# - scapy: for debugging and mocking network packets for tests
RUN python3 -m pip install --no-cache-dir \
        pytest \
        pyyaml \
        scapy

# ==============================
# DOCA Target
# This stage is only built when --target doca is specified. It contains any DOCA-specific configurations.
# ==============================
FROM common-deps AS doca
# DOCA-specific installation or steps (if needed)

# ==============================
# DPDK Target
# This stage is only built when --target dpdk is specified. It contains any DPDK-specific configurations.
# ==============================
FROM common-deps AS dpdk
# DPDK-specific installation or steps (if needed)

# ==============================
# Rivermax Target
# This stage is only built when --target rivermax is specified. It installs and configures Rivermax SDK.
# ==============================
FROM common-deps AS rivermax

# Define Rivermax-specific build arguments and environment variables
ARG RIVERMAX_VERSION=1.71.4
ARG RIVERMAX_SDK_ZIP_PATH=./rivermax_ubuntu2204_${RIVERMAX_VERSION}.tar.gz
ARG MAXPROC=8


# RMAX_TEGRA controls whether the build targets NVIDIA's Tegra platform (default is OFF).
# Affects Rivermax sample apps (via build args) and HoloHub (via environment variable).
# Set to ON/TRUE to target Tegra devices like Jetson.
ARG RMAX_TEGRA=OFF
ENV RMAX_TEGRA=${RMAX_TEGRA}

# Install additional dependencies required for Rivermax
# Update package cache and install additional dependencies required for Rivermax
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    libcap-dev \
    gdb \
    ethtool \
    iputils-ping \
    net-tools \
    libfmt-dev \
    libnl-3-dev \
    libnl-genl-3-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libglew-dev \
    cuda-nvml-dev-12-6 \
    cuda-nvtx-12-6 \
    && rm -rf /var/lib/apt/lists/*

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

# Build rmax_apps_lib sample applications
RUN cd rivermax-dev-kit && \
    cmake -B build -DRMAX_CUDA=ON -DRMAX_TEGRA=${RMAX_TEGRA} -DRMAX_BUILD_VIEWER=ON && \
    cmake --build build -j $(nproc)

# ==============================
# Default stage: common-deps
# If no target is specified, the common-deps stage will be built by default.
# ==============================
FROM common-deps
