# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Use base image from Holoscan ######################################################
# Requires Holoscan dev container to be built first.
#####################################################################################
# Args used in FROM commands must come first
# ============ Stage: base ============
# Configure the base environment
ARG FROM_IMAGE="holoscan-sdk-dev"
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.5.0 AS base

# End base image

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME
ENV CUPY_CACHE_DIR=./.cupy

# Install nice-to-haves
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      sudo git vim build-essential libtool autoconf unzip make wget curl \
      cmake iputils-ping python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install pillow image processing library for python
RUN pip install pillow


# Create 'docker' user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 \
            --ingroup docker \
            --home /home/docker \
            --shell /bin/bash \
            --disabled-password \
            --gecos "" docker
RUN echo "docker:docker" | chpasswd && adduser docker sudo

# Install fixuid
ARG FIXUID_URL="https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz"
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL ${FIXUID_URL} | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml && \
    echo "docker ALL = (root) NOPASSWD: ALL" >> /etc/sudoers

# Set entrypoint
USER docker:docker
WORKDIR /home/docker
