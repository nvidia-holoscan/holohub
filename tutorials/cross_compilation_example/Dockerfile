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

ARG CROSS_BASE_IMAGE=ubuntu:24.04

# Reject unsupported build hosts before installing the cross-compilation dependencies.
FROM ${CROSS_BASE_IMAGE} AS build-host

ARG BUILDARCH

RUN if [ "${BUILDARCH}" != "amd64" ]; then \
    echo "Cross compilation requires an x86_64/amd64 build host." >&2; \
    exit 1; \
  fi

# Install the pinned native build tools, Ubuntu AArch64 compiler/sysroot, and CMake release.
FROM build-host AS build-tools

ARG DEBIAN_FRONTEND=noninteractive

RUN test "$(dpkg --print-architecture)" = amd64 \
  && apt-get update \
  && apt-get install --yes --no-install-recommends \
  ca-certificates=20260601~24.04.1 \
  curl=8.5.0-2ubuntu10.13 \
  file=1:5.45-3build1 \
  g++-aarch64-linux-gnu=4:13.2.0-7ubuntu1 \
  gnupg=2.4.4-2ubuntu17.6 \
  ninja-build=1.11.1-2 \
  python3-pip=24.0+dfsg-1ubuntu1.3 \
  && rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=4.0.3

RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir \
  "cmake==${CMAKE_VERSION}" \
  && cmake --version | grep -q "cmake version ${CMAKE_VERSION}"

# Configure the NVIDIA x86_64, Cross-SBSA, and SBSA package repositories.
FROM build-tools AS cuda-repositories

RUN curl -fsSL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | gpg --dearmor --yes --output /usr/share/keyrings/cuda-archive-keyring.gpg \
  && printf 'deb [signed-by=%s] %s/ /\n' \
  /usr/share/keyrings/cuda-archive-keyring.gpg \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 \
  > /etc/apt/sources.list.d/nvidia-cuda-x86_64.list \
  && printf 'deb [signed-by=%s] %s/ /\n' \
  /usr/share/keyrings/cuda-archive-keyring.gpg \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa \
  > /etc/apt/sources.list.d/nvidia-cuda-cross-sbsa.list \
  && printf 'deb [arch=arm64 signed-by=%s] %s/ /\n' \
  /usr/share/keyrings/cuda-archive-keyring.gpg \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa \
  > /etc/apt/sources.list.d/nvidia-holoscan-sbsa.list

# Install native NVCC and the AArch64 CUDA cross-compilation packages.
FROM cuda-repositories AS cuda-cross

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
  cuda-cross-sbsa-13-0=13.0.3-1 \
  cuda-nvcc-13-0=13.0.88-1 \
  && rm -rf /var/lib/apt/lists/*

# Download, validate, and extract the AArch64 Holoscan SDK package.
FROM cuda-cross AS holoscan-sdk

ARG HOLOSCAN_DEB_VERSION=4.6.0.0-1

RUN apt-get update \
  -o Dir::Etc::sourcelist=sources.list.d/nvidia-holoscan-sbsa.list \
  -o Dir::Etc::sourceparts=- \
  -o APT::Architecture=arm64 \
  -o APT::Architectures=arm64 \
  && cd /tmp \
  && apt-get download \
  -o APT::Architecture=arm64 \
  -o APT::Architectures=arm64 \
  "holoscan-cuda-13:arm64=${HOLOSCAN_DEB_VERSION}" \
  && test "$(dpkg-deb -f holoscan-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb Architecture)" = arm64 \
  && test "$(dpkg-deb -f holoscan-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb Version)" = "${HOLOSCAN_DEB_VERSION}" \
  && dpkg-deb --extract holoscan-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb / \
  && test -f /opt/nvidia/holoscan/include/holoscan/holoscan.hpp \
  && test -f /opt/nvidia/holoscan/lib/cmake/holoscan/holoscan-config.cmake \
  && file -L /opt/nvidia/holoscan/lib/libholoscan_core.so | grep -q 'ARM aarch64' \
  && rm -f holoscan-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb \
  && rm -rf /var/lib/apt/lists/*

# Install and verify the standalone Holoscan CLI used by the containerized build flow.
FROM holoscan-sdk AS holoscan-cli

ARG HOLOSCAN_CLI_VERSION=4.6.0

RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir \
  --extra-index-url https://pypi.nvidia.com \
  "holoscan-cli==${HOLOSCAN_CLI_VERSION}" \
  && holoscan version --json | grep -q '"version": "'"${HOLOSCAN_CLI_VERSION}"'"'

# Add the cross toolchain files and define the final workspace.
FROM holoscan-cli AS cross-builder

COPY tutorials/cross_compilation_example/cmake/aarch64-cross-sbsa.cmake /opt/holoscan-cross/
COPY --chmod=755 tutorials/cross_compilation_example/cmake/nvcc-cross-sbsa /opt/holoscan-cross/

WORKDIR /workspace
