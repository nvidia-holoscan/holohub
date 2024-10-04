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

ARG BASE_IMAGE
ARG GPU_TYPE
############################################################
# Torch image (Used to extract the ARM64 dGPU pytorch wheel)
############################################################
FROM nvcr.io/nvidia/pytorch:23.10-py3 as pytorch

############################################################
# Base image
############################################################
FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND=noninteractive
#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Install PortAudio to enable PyAudio
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy over app-specific requirements
COPY applications/holoscrub/lmm/requirements.txt /tmp/requirements.txt

# Install setuptools prior to all other requirements to avoid install errors
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Clone Llama.cpp and checkout a stable commit
RUN git clone https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    # Commit from 01/05/24
    && git checkout be36bb946a6336238e92706464de6a30495fe825 \
    && mkdir build \
    && cd build \
    && cmake .. -DLLAMA_CUBLAS=ON \
    && LLAMA_CUDA_F16=true cmake --build . --config Release

WORKDIR /tmp/pip/

WORKDIR /workspace/holohub
