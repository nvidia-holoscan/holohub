# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# Set up prerequisites to run HoloHub CLI
#
# --------------------------------------------------------------------------
FROM base AS holohub-cli-prerequisites

# Install python3 if not present (needed for holohub CLI)
ARG PYTHON_VERSION=python3
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            software-properties-common curl gpg-agent \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} \
        && apt purge -y \
            python3-pip \
            software-properties-common \
        && apt-get autoremove --purge -y \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON_VERSION} 100 \
        && if [ "${PYTHON_VERSION}" != "python3" ]; then \
            update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_VERSION} 100 \
            ; fi \
    ; fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! python3 -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_VERSION} \
    ; fi

# --------------------------------------------------------------------------
#
# Use HoloHub CLI to set up common packages for developing with Holoscan SDK
#
# --------------------------------------------------------------------------
FROM holohub-cli-prerequisites AS holohub-cli

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
#
# Tracks2Endo4D application dependencies
#
# --------------------------------------------------------------------------
FROM holohub-cli AS application-dependencies

# Set the working directory
WORKDIR /tmp

RUN git clone --depth 1 https://github.com/deepmind/tapnet.git

# Override tapnext modules with custom versions before installing
COPY applications/tracks2endo4d/onnx_conversion/tapnext_torch.py tapnet/tapnet/tapnext/tapnext_torch.py
COPY applications/tracks2endo4d/onnx_conversion/tapnext_lru_modules.py tapnet/tapnet/tapnext/tapnext_lru_modules.py

# Override evaluation_datasets.py to avoid tensorflow import error
RUN echo "" > tapnet/tapnet/tapvid/evaluation_datasets.py

RUN cd tapnet && pip install .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install onnx onnxruntime numpy einops onnxscript

RUN mkdir -p checkpoints \
    && wget --progress=dot:giga --tries=3 --timeout=300 \
       https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz \
       -O checkpoints/bootstapnext_ckpt.npz

# Copy the ONNX tapnext conversion script to the builder stage and run the conversion
COPY applications/tracks2endo4d/onnx_conversion/onnx_tapnext.py .

RUN python onnx_tapnext.py --checkpoint checkpoints/bootstapnext_ckpt.npz --output_path /models/ --device cpu

# Copy only this application (context is HoloHub root)
COPY applications/tracks2endo4d /workspace/app

# Set the working directory
WORKDIR /workspace
