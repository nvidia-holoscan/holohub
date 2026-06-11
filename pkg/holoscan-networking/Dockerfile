# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG HOLOSCAN_DEB_REMOTE_VERSION=4.3.0 # only used if HOLOSCAN_DEB_SRC is "remote"
ARG HOLOSCAN_DEB_LOCAL_PATH="./holoscan.deb" # only used if HOLOSCAN_DEB_SRC is "local"
ARG DAQIRI_REPO="https://github.com/NVIDIA/daqiri.git"
ARG DAQIRI_REF="main"
ARG DAQIRI_MGR="dpdk socket rdma"
ARG DAQIRI_BUILD_PYTHON=ON
ARG DAQIRI_BUILD_EXAMPLES=OFF
ARG DAQIRI_CUDA_ARCHITECTURES="80;89;90"
ARG BUILD_SHARED_LIBS=ON
ARG DAQIRI_INSTALL_PREFIX=/opt/daqiri

# ============================================================
# base: Base layer
# ============================================================
FROM nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04 AS base
SHELL ["/bin/bash", "-eou", "pipefail", "-c"]

# Basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        infiniband-diags \
        mft \
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
        cmake="3.31.11*" \
        cmake-data="3.31.11*" \
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
# base-deps: base dependencies for DAQIRI networking examples
# ==============================================================
FROM hsdk-installed AS base-deps

ARG HOLOSCAN_DEB_REMOTE_VERSION
ARG TARGETARCH
ARG CACHEBUST=1
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# APT installs for DAQIRI base dependencies
# - libibverbs-dev, librdmacm-dev: RDMA/ibverbs support for Mellanox NICs
# - libmlx5-1: Mellanox ConnectX driver
# - ibverbs-utils: utilities
# - python3-dev: for building python bindings
# - cargo, rustc: Rust bridge build used by psd_pipeline packetizer
RUN CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p' | sed 's/\./-/g') \
    && test -n "${CUDA_VERSION}" \
    && apt-get update && apt-get install -y --no-install-recommends \
        cargo \
        libibverbs-dev \
        libcublas-dev-${CUDA_VERSION} \
        libcufft-dev-${CUDA_VERSION} \
        libcurand-dev-${CUDA_VERSION} \
        libcusolver-dev-${CUDA_VERSION} \
        libcusparse-dev-${CUDA_VERSION} \
        libvulkan-dev \
        librdmacm-dev \
        libmlx5-1 \
        ibverbs-utils \
        python3-dev \
        rustc \
    && rm -rf /var/lib/apt/lists/*

# PIP installs
# - holoscan: Python SDK bindings for Python HoloHub examples
# - pytest: test harness
# - pyyaml: to parse yaml configs in tests
# - scapy: for debugging and mocking network packets for tests
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed pip \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        holoscan==${HOLOSCAN_DEB_REMOTE_VERSION} \
        pybind11 \
        pytest \
        pyyaml \
        scapy

ENV LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib:/usr/local/lib/python3.12/dist-packages/holoscan/lib:${LD_LIBRARY_PATH}

# Finalize the wheel_axle-managed Holoscan Python wheel while still root so
# non-root HoloHub container users do not need to write to site-packages.
RUN python3 -c "import holoscan"

# ==============================================================
# dpdk-deps: Build upstream DPDK from source
# ==============================================================
FROM base-deps AS dpdk

# DPDK version to download and build
ARG DPDK_VERSION=25.11
ARG DPDK_BUILD_DIR=/opt/dpdk-build
ARG DPDK_INSTALL_PREFIX=/usr/local


# - infiniband-diags for IB diagnostics
# - mft for Mellanox Flex Transport
# - ninja-build: for cmake build
# - pkgconf: to import dpdk in CMake
# - meson: for building DPDK
# - python3-pyelftools: required for DPDK build
# - libnuma-dev: NUMA support for DPDK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build \
        pkgconf \
        meson \
        python3-pyelftools \
        libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pkg/holoscan-networking/dpdk_patches /tmp/dpdk_patches

# Download, build and install DPDK using meson/ninja
RUN curl -fsSL https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz -o /tmp/dpdk-${DPDK_VERSION}.tar.xz \
    && tar xf /tmp/dpdk-${DPDK_VERSION}.tar.xz -C /tmp \
    && cd /tmp/dpdk-${DPDK_VERSION} \
    && for patch_file in /tmp/dpdk_patches/*.patch; do \
         patch_name="$(basename "${patch_file}")"; \
         echo "Applying DPDK patch: ${patch_name}"; \
         if [[ "${patch_name}" == "dmabuf.patch" ]]; then \
           git apply \
             --exclude=.mailmap \
             --exclude=doc/guides/rel_notes/release_26_03.rst \
             "${patch_file}"; \
         else \
           git apply "${patch_file}"; \
         fi; \
       done \
    && meson setup ${DPDK_BUILD_DIR} \
        --prefix=${DPDK_INSTALL_PREFIX} \
        -Dtests=false \
        -Dplatform=generic \
        -Denable_docs=false \
        -Ddisable_apps=* \
        -Denable_drivers=bus/auxiliary,bus/pci,bus/vdev,common/mlx5,mempool/ring,mempool/stack,net/af_packet,net/mlx5,net/ring \
    && cd ${DPDK_BUILD_DIR} \
    && ninja \
    && meson install \
    && ldconfig \
    && rm -rf /tmp/dpdk-${DPDK_VERSION} /tmp/dpdk-${DPDK_VERSION}.tar.xz ${DPDK_BUILD_DIR} /tmp/dpdk_patches

# ==============================================================
# rdma: Named target for consistent per-manager container builds.
# Identical to dpdk (which already includes RDMA/ibverbs deps).
# ==============================================================
FROM dpdk AS rdma

# ==============================================================
# daqiri-package: build DAQIRI using the same manager stack as
# NVIDIA/daqiri's Dockerfile, then wrap the install tree in a deb.
# ==============================================================
FROM rdma AS daqiri-package

ARG DAQIRI_REPO
ARG DAQIRI_REF
ARG DAQIRI_MGR
ARG DAQIRI_BUILD_PYTHON
ARG DAQIRI_BUILD_EXAMPLES
ARG DAQIRI_CUDA_ARCHITECTURES
ARG BUILD_SHARED_LIBS
ARG DAQIRI_INSTALL_PREFIX

ENV pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11

RUN git clone "${DAQIRI_REPO}" /tmp/daqiri-src \
    && cd /tmp/daqiri-src \
    && git checkout "${DAQIRI_REF}" \
    && git submodule update --init --recursive \
    && sed -i "s/^set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES \"${DAQIRI_CUDA_ARCHITECTURES}\")/" src/CMakeLists.txt \
    && DAQIRI_VERSION="$(sed -n 's/^project(daqiri VERSION \([^ ]*\).*$/\1/p' CMakeLists.txt)" \
    && test -n "${DAQIRI_VERSION}" \
    && cmake -S /tmp/daqiri-src -B /tmp/daqiri-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${DAQIRI_INSTALL_PREFIX}" \
        -DCMAKE_CUDA_ARCHITECTURES="${DAQIRI_CUDA_ARCHITECTURES}" \
        -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        -DDAQIRI_BUILD_EXAMPLES="${DAQIRI_BUILD_EXAMPLES}" \
        -DDAQIRI_BUILD_PYTHON="${DAQIRI_BUILD_PYTHON}" \
        -DDAQIRI_MGR="${DAQIRI_MGR}" \
    && cmake --build /tmp/daqiri-build -j "$(nproc)" \
    && DESTDIR=/tmp/daqiri-package-root cmake --install /tmp/daqiri-build \
    && mkdir -p /tmp/daqiri-package-root/DEBIAN \
    && { \
        echo "Package: daqiri"; \
        echo "Version: ${DAQIRI_VERSION}"; \
        echo "Section: libs"; \
        echo "Priority: optional"; \
        echo "Architecture: $(dpkg --print-architecture)"; \
        echo "Maintainer: NVIDIA <holoscan-support@nvidia.com>"; \
        echo "Description: Data Acquisition for Integrated Real-time Instruments"; \
        echo " DAQIRI networking runtime, development files, and optional Python bindings."; \
      } > /tmp/daqiri-package-root/DEBIAN/control \
    && dpkg-deb --build /tmp/daqiri-package-root /tmp/daqiri.deb \
    && rm -rf /tmp/daqiri-src /tmp/daqiri-build /tmp/daqiri-package-root

# ==============================
# Default stage: dpdk with DAQIRI
# If no target is specified, this stage will be built by default.
# ==============================
FROM dpdk AS runtime

ARG DAQIRI_INSTALL_PREFIX

COPY --from=daqiri-package /tmp/daqiri.deb /tmp/daqiri.deb
RUN apt-get update \
    && apt-get install -y --no-install-recommends /tmp/daqiri.deb \
    && rm -f /tmp/daqiri.deb \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# DAQIRI's Python bindings use pybind11's CMake package. The package is
# installed with pip above, so expose that package directory to CMake.
ENV pybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11
ENV DAQIRI_ROOT=${DAQIRI_INSTALL_PREFIX}
ENV CMAKE_PREFIX_PATH=${DAQIRI_INSTALL_PREFIX}:/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11
ENV LD_LIBRARY_PATH=${DAQIRI_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${DAQIRI_INSTALL_PREFIX}/lib/python

FROM runtime
