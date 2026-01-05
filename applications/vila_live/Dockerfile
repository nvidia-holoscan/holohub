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

FROM ${BASE_IMAGE} AS base

# ------------------------------------------------------------------------
# Holohub setup
# ------------------------------------------------------------------------
RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub && \
    /tmp/scripts/holohub setup && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install setup tools and PyTorch
# ----------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir setuptools packaging && \
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ----------------------------------------------------------------------------
# Install flash-attention 2
# ----------------------------------------------------------------------------
# NOTE: on Arm64 this will take between 1-1.5 hours
# NOTE: MAX_JOBS is set to 4 to avoid running out of memory
RUN MAX_JOBS=4 python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.7.*

# ----------------------------------------------------------------------------
# Clone and install AWQ based on the GPU's compute capacity
# ----------------------------------------------------------------------------
# Copy over patch file to patch llm-awq so that it works with transformers 4.48
COPY applications/vila_live/llm-awq.patch /tmp/

# Clone and install AWQ based on the GPU's compute capacity
WORKDIR /workspace
ARG COMPUTE_CAPACITY
RUN git clone https://github.com/mit-han-lab/llm-awq && \
    cd llm-awq && \
    git checkout 41d7e704932b7eaf8b3dcfb21e1fe49fd296d894 && \
    # patching this version of llm-awq so that it works with transformers 4.48
    git apply /tmp/llm-awq.patch && \
    # Remove torch and transformers from pyproject.toml to avoid conflicts with installed versions
    sed -i '/torch/d' pyproject.toml && \
    sed -i '/transformers/d' pyproject.toml && \
    python3 -m pip install --no-cache-dir -e . && \
    cd awq/kernels && \
    TORCH_CUDA_ARCH_LIST="${COMPUTE_CAPACITY}+PTX" MAX_JOBS=$(nproc) python setup.py install

# ----------------------------------------------------------------------------
# Clone and install VILA repository
# ----------------------------------------------------------------------------
RUN git clone https://github.com/Efficient-Large-Model/VILA.git && \
    cd VILA && \
    git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5 && \
    # Remove torch and transformers from pyproject.toml to avoid conflicts with installed versions
    sed -i '/torch/d' pyproject.toml && \
    sed -i '/transformers/d' pyproject.toml && \
    python3 -m pip install --no-cache-dir --no-deps -e .

# ----------------------------------------------------------------------------
# Install Application-specific dependencies
# ----------------------------------------------------------------------------
# Copy over app-specific requirements
COPY applications/vila_live/requirements.txt /tmp/
# Install requirements
RUN python3 -m pip install -r /tmp/requirements.txt

# ----------------------------------------------------------------------------
# Download the AWQ-quantized VILA model
# ----------------------------------------------------------------------------
RUN mkdir -p /workspace/volumes/models && \
    huggingface-cli download Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix-AWQ \
    --local-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-Fix-AWQ \
    --cache-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-Fix-AWQ

# ----------------------------------------------------------------------------
# Set up final working directory to be the holohub directory
# ----------------------------------------------------------------------------
WORKDIR /workspace/holohub
