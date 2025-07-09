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

ARG BASE_IMAGE
ARG GPU_TYPE
ARG COMPUTE_CAPACITY
############################################################
# Torch image (Used to extract the ARM64 dGPU pytorch wheel)
############################################################
FROM nvcr.io/nvidia/pytorch:24.06-py3 as pytorch

############################################################
# Base image
############################################################
FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND=noninteractive

# Holohub dev setup
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set up Git LFS
RUN git lfs install

# Copy over app-specific requirements
COPY applications/sam2/requirements.txt /tmp/requirements.txt

# Copy the Torch wheel from the PyTorch image. This is necessary because the PyTorch wheel is not available for ARM64 on PyPI.
WORKDIR /tmp/pip/
# enter a bash shell to debug
RUN /bin/bash
# Install the Torch wheel based on the target architecture
# ARG TARGETARCH
# RUN if [ "$TARGETARCH" = "amd64" ]; then \
#         echo "Building for x86 (AMD64) architecture"; \
#         python3 -m pip install torch>=2.3.1; \
#     elif [ "$TARGETARCH" = "arm64" ]; then \
#         echo "Not tested on ARM - Building for ARM64 architecture"; \
#         python3 -m pip install torch>=2.3.1; \
#     else \
#         echo "Unknown architecture: $TARGETARCH"; \
#         exit 1; \
#     fi

# Clone sam2
WORKDIR /workspace
ARG COMPUTE_CAPACITY
RUN git clone https://github.com/facebookresearch/segment-anything-2.git \
    && cd segment-anything-2 \
    && python3 -m pip install --no-cache-dir -e . \
    && python3 -m pip install --no-cache-dir -e ".[demo]" \
    && cd checkpoints \
    && ./download_ckpts.sh

# Install setuptools prior to all other requirements to avoid install errors
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/holohub
