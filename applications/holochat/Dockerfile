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


# --------------------------------------------------------------------------
# Base image
# --------------------------------------------------------------------------
ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

# --------------------------------------------------------------------------
# Holohub setup
# --------------------------------------------------------------------------
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# Set default Holohub data directory
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

# --------------------------------------------------------------------------
# Install PyPI Dependencies
# --------------------------------------------------------------------------
COPY applications/holochat/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# --------------------------------------------------------------------------
# Clone and build Llama.cpp
# --------------------------------------------------------------------------
# Clone Llama.cpp (at a specific commit to avoid breaking changes)
WORKDIR /workspace
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git checkout 4b2a4778f81f222c12271ce3b1997990b3071faf

# Build Llama.cpp (with CUDA support)
WORKDIR /workspace/llama.cpp
RUN cmake -B build \
        -DGGML_CUDA:BOOL=ON \
        -DGGML_CUDA_FORCE_CUBLAS:BOOL=ON \
        -DLLAMA_CURL:BOOL=OFF \
    && cmake --build build --config Release

# Set working directory back to holohub
WORKDIR /workspace/holohub
