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

FROM nvcr.io/nvidia/cuda:12.9.0-base-ubuntu24.04

ARG BASE_SDK_VERSION
# Install dependencies
RUN apt-get update && apt-get install -y \
    # Need numpy for converting video to gxf entity
    python3-pip \
    python3-numpy \
    cmake \
    git \
    ffmpeg \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && curl -S -# -L -o ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && chmod u+x ngc-cli/ngc \
    && ln -s $(pwd)/ngc-cli/ngc /usr/local/bin/

# Install Holoscan SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-9 \
    cuda-nvrtc-dev-12-9 \
    holoscan=${BASE_SDK_VERSION}.* \
    libnvinfer-headers-dev=*cuda12.9* \
    libnvinfer10=*cuda12.9* \
    libnvinfer-dev=*cuda12.9* \
    libnvinfer-headers-plugin-dev=*cuda12.9* \
    libnvinfer-plugin10=*cuda12.9* \
    libnvinfer-plugin-dev=*cuda12.9* \
    libnvonnxparsers10=*cuda12.9* \
    libnvonnxparsers-dev=*cuda12.9* \
    libnpp-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*
