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
# Base image
############################################################
FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND=noninteractive

# Holohub setup
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Copy over app-specific requirements
COPY applications/vila_live/requirements.txt /tmp/requirements.txt

# Install setuptools prior to all other requirements to avoid install errors
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Build/install flash-attention 2 (NOTE: on Arm64 this will take between 1-1.5 hours)
# MAX_JOBS is set to 4 to avoid running out of memory
RUN MAX_JOBS=4 python3 -m pip install --no-cache-dir --no-build-isolation flash-attn~=2.5

# Clone AWQ and VILA repositories and install them based on the GPU's compute capacity
WORKDIR /workspace
ARG COMPUTE_CAPACITY
RUN git clone https://github.com/mit-han-lab/llm-awq \
    && cd llm-awq \
    && git checkout 41d7e704932b7eaf8b3dcfb21e1fe49fd296d894 \
    # Remove torch and transformers from pyproject.toml to avoid conflicts with installed versions
    && sed -i '/torch/d' pyproject.toml \
    && sed -i '/transformers/d' pyproject.toml \
    && python3 -m pip install --no-cache-dir -e . \
    && cd awq/kernels \
    && TORCH_CUDA_ARCH_LIST="${COMPUTE_CAPACITY}+PTX" \
    MAX_JOBS=$(nproc) python setup.py install \
    && cd ../.. \
    && git clone https://github.com/Efficient-Large-Model/VILA.git \
    && cd VILA \
    && git checkout ec7fb2c264920bf004fd9fa37f1ec36ea0942db5 \
    # Remove torch and transformers from pyproject.toml to avoid conflicts with installed versions
    && sed -i '/torch/d' pyproject.toml \
    && sed -i '/transformers/d' pyproject.toml \
    && python3 -m pip install --no-cache-dir --no-deps -e .

# Download the AWQ-quantized VILA model
RUN huggingface-cli download Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ \
    --local-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ --cache-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ

WORKDIR /workspace/holohub