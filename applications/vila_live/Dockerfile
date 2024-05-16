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
FROM nvcr.io/nvidia/pytorch:23.10-py3 as pytorch

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
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the Torch wheel from the PyTorch image. This is necessary because the PyTorch wheel is not available for ARM64 on PyPI.
WORKDIR /tmp/pip/
COPY --from=pytorch /tmp/pip/torch*.whl /tmp/pip/

# Install the Torch wheel based on the target architecture
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        echo "Building for x86 (AMD64) architecture"; \
        python3 -m pip install torch==2.1.0; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        echo "Building for ARM64 architecture"; \
        python3 -m pip install /tmp/pip/torch-2.1.0a0+32f93b1-cp310-cp310-linux_aarch64.whl; \
    else \
        echo "Unknown architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Build/install flash-attention 2 (NOTE: on Arm64 this will take between 1-1.5 hours)
# MAX_JOBS is set to 4 to avoid running out of memory
RUN MAX_JOBS=4 python3 -m pip install --no-cache-dir --no-build-isolation flash-attn

# Clone AWQ and VILA repositories and install them based on the GPU's compute capacity
WORKDIR /workspace
ARG COMPUTE_CAPACITY
RUN git clone https://github.com/mit-han-lab/llm-awq \
    && cd llm-awq \
    && git checkout 41d7e704932b7eaf8b3dcfb21e1fe49fd296d894 \
    # Remove torch from pyproject.toml to avoid conflicts with the installed torch version
    && sed -i '/torch/d' pyproject.toml \
    && python3 -m pip install --no-cache-dir -e . \
    && cd awq/kernels \
    && TORCH_CUDA_ARCH_LIST="${COMPUTE_CAPACITY}+PTX" \
    MAX_JOBS=$(nproc) python setup.py install \
    && cd ../.. \
    && git clone https://github.com/Efficient-Large-Model/VILA.git \
    && cd VILA \
    && git checkout d7d54bc4ca1e582f59516ba2f94a0217ad2430a0 \
    # Remove torch from pyproject.toml to avoid conflicts with the installed torch version
    && sed -i '/torch/d' pyproject.toml \
    && python3 -m pip install --no-cache-dir --no-deps -e .

# Download the AWQ-quantized VILA model
RUN huggingface-cli download Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ \
    --local-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ --cache-dir /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ

WORKDIR /workspace/holohub
