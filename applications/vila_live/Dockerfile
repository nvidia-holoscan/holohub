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

ARG BASE_IMAGE
ARG GPU_TYPE
ARG COMPUTE_CAPACITY
ARG CUDA_MAJOR

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
ARG CUDA_MAJOR
RUN python3 -m pip install --no-cache-dir setuptools packaging && \
    if [ "${CUDA_MAJOR}" = "13" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cu130"; \
    else \
        INDEX_URL="https://download.pytorch.org/whl/cu124"; \
    fi && \
    python3 -m pip install --no-cache-dir torch torchvision --index-url ${INDEX_URL}

# ----------------------------------------------------------------------------
# Install flash-attention 2
# ----------------------------------------------------------------------------
# NOTE: on Arm64 this will take between 1-1.5 hours
# NOTE: MAX_JOBS is set to 4 to avoid running out of memory
# NOTE: CUDA 13 on x86_64 uses prebuilt wheels from https://github.com/mjun0812/flash-attention-prebuild-wheels
ARG CUDA_MAJOR
RUN if [ "${CUDA_MAJOR}" = "13" ] && [ "$(uname -m)" = "x86_64" ]; then \
        PYTHON_VERSION=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") && \
        TORCH_VERSION=$(python3 -c "import torch; print(f'torch{torch.__version__.split(\".\")[0]}.{torch.__version__.split(\".\")[1]}')") && \
        WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130${TORCH_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl" && \
        echo "Installing flash-attn wheel for CUDA 13 (x86_64): ${WHEEL_URL}" && \
        python3 -m pip install --no-cache-dir "${WHEEL_URL}"; \
    else \
        MAX_JOBS=4 python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.7.*; \
    fi

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
