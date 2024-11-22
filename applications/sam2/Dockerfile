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


FROM nvcr.io/nvidia/pytorch:24.06-py3 AS pytorch


# Install dependencies
RUN apt-get update && apt-get install -y \
    git \   
    git-lfs \ 
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Set up Git LFS
RUN git lfs install


# Clone sam2
WORKDIR /workspace
ARG COMPUTE_CAPACITY
RUN git clone https://github.com/facebookresearch/sam2.git \
    && cd sam2 \
    && python3 -m pip install --no-cache-dir -e . \
    && python3 -m pip install --no-cache-dir -e ".[demo]" \
    && cd checkpoints \
    && ./download_ckpts.sh
WORKDIR /workspace


# Check the architecture and download the CUDA keyring
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64 \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb \
    ; elif [ $(uname -m) = "x86_64" ]; then ARCH=x86_64 \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    ; else echo "Unsupported architecture"; fi
RUN dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get -y install holoscan


# Setup Docker & NVIDIA Container Toolkit's apt repositories to enable DooD
# for packaging & running applications with the CLI
# Ref: Docker installation: https://docs.docker.com/engine/install/ubuntu/
# DooD (Docker-out-of-Docker): use the Docker (or Moby) CLI in your dev container to connect to
#  your host's Docker daemon by bind mounting the Docker Unix socket.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        valgrind="1:3.18.1-*" \
        xvfb="2:21.1.4-*" \
        libx11-dev="2:1.7.5-*" \
        libxcb-glx0="1.14-*" \
        libxcursor-dev="1:1.2.0-*" \
        libxi-dev="2:1.8-*" \
        libxinerama-dev="2:1.1.4-*" \
        libxrandr-dev="2:1.5.2-*" \
        libvulkan-dev="1.3.204.1-*" \
        glslang-tools="11.8.0+1.3.204.0-*" \
        vulkan-validationlayers="1.3.204.1-*" \
        libwayland-dev="1.20.0-*" \
        libxkbcommon-dev="1.4.0-*" \
        pkg-config="0.29.2-*" \
        libdecor-0-plugin-1-cairo="0.1.0-*" \
        libegl1="1.4.0-*" \
        libopenblas0="0.3.20+ds-*" \
        libv4l-dev="1.22.1-*" \
        v4l-utils="1.22.1-*" \
        libpng-dev="1.6.37-*" \
        libjpeg-turbo8-dev="2.1.2-*" \
        docker-ce-cli="5:25.0.3-*" \
        docker-buildx-plugin="0.12.1-*" \
    && rm -rf /var/lib/apt/lists/*
# start a bash shell to debug
RUN /bin/bash
