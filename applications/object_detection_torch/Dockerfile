# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

FROM base as torch

# Install libraries
ARG GPU_TYPE
RUN if [ "${GPU_TYPE}" = "igpu" ]; then \
        PYTORCH_WHEEL_VERSION="2.8.0"; \
        TORCHVISION_WHEEL_VERSION="0.23.0"; \
        INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu129"; \
    else \
        CUDA_MAJOR=$(nvcc --version | grep -o "release [0-9]*" | awk '{print $2}'); \
        if [ "$CUDA_MAJOR" = "13" ]; then \
            # Reflect observed Torchvision wheel dependencies for CUDA 13
            PYTORCH_WHEEL_VERSION="2.9.0+cu130"; \
            TORCHVISION_WHEEL_VERSION="0.24.0"; \
            INDEX_URL="https://download.pytorch.org/whl/"; \
        else \
            PYTORCH_WHEEL_VERSION="2.8.0+cu129"; \
            TORCHVISION_WHEEL_VERSION="0.23.0+cu129"; \
            INDEX_URL="https://download.pytorch.org/whl/"; \
        fi; \
    fi; \
    echo "Installing torch==${PYTORCH_WHEEL_VERSION} from $INDEX_URL"; \
    python3 -m pip install --force-reinstall torch==${PYTORCH_WHEEL_VERSION} torchvision==${TORCHVISION_WHEEL_VERSION} torchaudio --index-url $INDEX_URL; \
    if ! find /usr/local/lib/python3.12/dist-packages/torch -name libtorch_cuda.so | grep -q .; then \
        echo "libtorch_cuda.so not found, torch installation failed"; \
        exit 1; \
    fi

RUN rm -rf /opt/libtorch/* && \
    mkdir -p /opt/libtorch && \
    LIBTORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])") && \
    ln -sf "${LIBTORCH_PATH}/lib" /opt/libtorch/lib


# Set up Holoscan SDK container libtorch to be found with ldconfig for app C++ build and runtime
RUN echo $(python3 -c "import torch; print(torch.__path__[0])")/lib > /etc/ld.so.conf.d/libtorch.conf \
    && ldconfig \
    && ldconfig -p | grep -q "libtorch.so"
