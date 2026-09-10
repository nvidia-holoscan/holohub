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

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=4.0.3
ARG HOLOSCAN_DEB_VERSION=4.6.0.0-1

RUN test "$(dpkg --print-architecture)" = amd64 \
  && apt-get update \
  && apt-get install --yes --no-install-recommends \
  ca-certificates \
  cmake \
  curl \
  file \
  g++-aarch64-linux-gnu \
  gnupg \
  ninja-build \
  python3-pip \
  && PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-cache-dir \
  "cmake==${CMAKE_VERSION}" \
  && cmake --version | grep -q "cmake version ${CMAKE_VERSION}" \
  && curl -fsSL \
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
  > /etc/apt/sources.list.d/nvidia-holoscan-sbsa.list \
  && apt-get update \
  && apt-get install --yes --no-install-recommends \
  cuda-cross-sbsa-13-0 \
  cuda-nvcc-13-0 \
  && apt-get update \
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

COPY --chmod=755 cmake/ /opt/holoscan-cross/

WORKDIR /workspace
