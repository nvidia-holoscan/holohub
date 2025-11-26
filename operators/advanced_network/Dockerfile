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
# base-deps: base dependencies for the advanced network lib
# ==============================================================
FROM hsdk-installed AS base-deps

ARG TARGETARCH
ARG CACHEBUST=1
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# APT installs for base dependencies (no DOCA/GPUNetIO packages)
# - cublas, cufft, cusolver, curand dev libs: for matx
# - ninja-build: for cmake build
# - pkgconf: to import dpdk in CMake
# - meson: for building DPDK
# - python3-pyelftools: required for DPDK build
# - libnuma-dev: NUMA support for DPDK
# - libibverbs-dev, librdmacm-dev: RDMA/ibverbs support for Mellanox NICs
# - libmlx5-1: Mellanox ConnectX driver
# - ibverbs-utils: utilities
# - python3-dev: for building python bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcublas-dev-12-6 \
        libcufft-dev-12-6 \
        libcusolver-dev-12-6 \
        libcurand-dev-12-6 \
        ninja-build \
        pkgconf \
        meson \
        python3-pyelftools \
        libnuma-dev \
        libibverbs-dev \
        librdmacm-dev \
        libmlx5-1 \
        ibverbs-utils \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# PIP installs
# - pytest: test harness
# - pyyaml: to parse yaml configs in tests
# - scapy: for debugging and mocking network packets for tests
RUN python3 -m pip install --no-cache-dir \
        pytest \
        pyyaml \
        scapy

# ==============================================================
# dpdk-build: Build upstream DPDK from source
# ==============================================================
FROM base-deps AS dpdk-build

ARG TARGETARCH
# DPDK version to download and build
ARG DPDK_VERSION=24.11.3
ARG DPDK_BUILD_DIR=/opt/dpdk-build
ARG DPDK_INSTALL_PREFIX=/usr/local
ARG DOCA_VERSION=2.8.0
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp

# Download DPDK source from official repository
ADD https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz /tmp/
RUN tar xf dpdk-${DPDK_VERSION}.tar.xz

# Build and install DPDK using meson/ninja
WORKDIR /tmp/dpdk-stable-${DPDK_VERSION}
RUN meson setup ${DPDK_BUILD_DIR} \
        --prefix=${DPDK_INSTALL_PREFIX} \
        -Denable_kmods=false \
        -Dtests=false \
        -Denable_docs=false \
        -Ddisable_drivers=baseband/*,bus/ifpga/*,common/cpt,common/dpaax,common/iavf,common/octeontx,common/octeontx2,crypto/nitrox,net/ark,net/atlantic,net/avp,net/axgbe,net/bnx2x,net/bnxt,net/cxgbe,net/e1000,net/ena,net/enic,net/fm10k,net/hinic,net/hns3,net/i40e,net/ixgbe,vdpa/ifc,net/igc,net/liquidio,net/netvsc,net/nfp,net/qede,net/sfc,net/thunderx,net/vdev_netvsc,net/vmxnet3,regex/octeontx2 \
    && cd ${DPDK_BUILD_DIR} \
    && ninja \
    && meson install \
    && ldconfig \
    && rm -rf /tmp/dpdk-stable-${DPDK_VERSION} /tmp/dpdk-stable-${DPDK_VERSION}.tar.xz ${DPDK_BUILD_DIR} /tmp/dpdk_patches

# Configure the DOCA APT repository for mlnx-ofed-kernel-utils
# (needed for ibdev2netdev utility used by tune_system.py)
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

# Install mlnx-ofed-kernel-utils (utilities including ibdev2netdev used by tune_system.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        mlnx-ofed-kernel-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ==============================
# DPDK Target (Default)
# This is the default target for users who only need DPDK support.
# GPUNetIO/DOCA is NOT included.
# ==============================
FROM dpdk-build AS dpdk
# DPDK is ready to use, no additional steps needed

# ==============================================================
# gpunetio-deps: Add DOCA SDK packages for GPUNetIO support
# ==============================================================
FROM dpdk-build AS gpunetio-deps

ARG DEBIAN_FRONTEND=noninteractive

# Install DOCA SDK packages required for GPUNetIO
# (DOCA repo is already configured in dpdk-build stage)
# - libdoca-sdk-dma-dev: for gpunetio backend (dependency of doca-gpunetio module)
# - libdoca-sdk-gpunetio-dev: for gpunetio backend (doca-gpunetio module)
# - libdoca-sdk-eth-dev: for gpunetio backend (doca-eth module)
# - libdoca-sdk-flow-dev: for gpunetio backend (doca-flow module)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libdoca-sdk-dma-dev \
        libdoca-sdk-gpunetio-dev \
        libdoca-sdk-eth-dev \
        libdoca-sdk-flow-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# GPUNetIO Target
# This stage is built when --target gpunetio is specified.
# It includes DOCA packages required for GPUNetIO support.
# ==============================
FROM gpunetio-deps AS gpunetio
# GPUNetIO is ready to use with DOCA support

# ==============================
# Rivermax Target
# This stage is only built when --target rivermax is specified. It installs and configures Rivermax SDK.
# Note: Rivermax does not require DPDK - extends from base-deps directly.
# ==============================
FROM base-deps AS rivermax

ARG TARGETARCH

# Define Rivermax-specific build arguments and environment variables
ARG RIVERMAX_VERSION=1.70.32
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
# Default stage: dpdk
# If no target is specified, the dpdk stage will be built by default.
# This provides upstream DPDK support without DOCA dependencies.
# Use --target gpunetio if you need DOCA GPUNetIO support.
# ==============================
FROM dpdk
