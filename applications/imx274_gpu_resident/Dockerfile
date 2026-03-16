# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =============================================================================
# IMX274 GPU-Resident Application Container (dgpu / aarch64)
# =============================================================================
#
# Builds a development container with Holoscan Sensor Bridge (HSB) and HoloHub
# CLI, starting from the Holoscan SDK base image.
#
# ---- Build and Run ----
#
#     ./holohub run-container imx274_gpu_resident
#
# =============================================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Display GPU-Resident Operator Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    libgbm-dev \
    libvulkan-dev \
    libdrm-dev \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    python3-dev \
    python3-setuptools \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libibverbs-dev \
    libnuma-dev \
    libelf-dev \
    libfmt-dev \
    libyaml-cpp-dev \
    libcurlpp-dev \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# HoloHub CLI Setup
# -----------------------------------------------------------------------------
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# -----------------------------------------------------------------------------
# Optional: DOCA SDK 3.2.1 with GPUNetIO (when INSTALL_DOCA=1, e.g. doca mode)
# -----------------------------------------------------------------------------
ARG INSTALL_DOCA=0
ARG TARGET_ARCH=arm64
RUN if [ "${INSTALL_DOCA}" = "1" ]; then \
        set -e && cd /opt && \
        if [ "${TARGET_ARCH}" = "amd64" ] || [ "${TARGET_ARCH}" = "x86_64" ]; then DOCA_ARCH=amd64; elif [ "${TARGET_ARCH}" = "arm64" ] || [ "${TARGET_ARCH}" = "aarch64" ]; then DOCA_ARCH=arm64; else echo "Unknown architecture: ${TARGET_ARCH}"; exit 1; fi && \
        UBUNTU_VERSION=$(grep "VERSION_ID" /etc/os-release | awk -F'=' '{print $2}' | sed 's/[."]//g') && \
        wget -q https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.1/host/doca-host_3.2.1-044000-25.10-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt install -o DPkg::Options::="--force-confnew" -y ./doca-host_3.2.1-044000-25.10-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt-get update \
        && rm -f ./doca-host_*.deb \
        && apt-get install -y --no-install-recommends ninja-build pkgconf mlnx-dpdk-dev libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev mlnx-ofed-kernel-utils ibverbs-utils python3-pyelftools \
        && rm -rf /var/lib/apt/lists/* \
        && sed -i '335s/(\*wqe)/(wqe)/' /opt/mellanox/doca/include/doca_gpunetio_dev_verbs_qp.cuh; \
    else true; fi

# Default working directory
WORKDIR /workspace/holohub

# HoloHub data path
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

# NVIDIA runtime capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display
ENV NVIDIA_VISIBLE_DEVICES=all
