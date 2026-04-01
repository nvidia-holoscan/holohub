# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# G-SHARP Surgical Scene Reconstruction
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
# G-SHARP application dependencies
# --------------------------------------------------------------------------
FROM holohub-cli AS application-dependencies

COPY applications/surgical_scene_recon/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# SAM3 imports decord at module level (sam3/train/data/sam3_image_dataset.py) but
# we only use SAM3 for image inference, never video decoding.  Install a tiny stub
# so the import succeeds on all architectures (decord has no ARM wheels).
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/decord && \
    printf '%s\n' \
      '"""Minimal decord stub - satisfies SAM3 import without native libs."""' \
      'def cpu(n=0): return n' \
      'class VideoReader:' \
      '    def __init__(self, *a, **kw): raise NotImplementedError("decord stub: video decoding not available")' \
      > /usr/local/lib/python3.12/dist-packages/decord/__init__.py

# gdown for downloading datasets from Google Drive (used by holoscan_download_data)
RUN pip install --no-cache-dir gdown

# Install VGGT and SAM3 from upstream (pinned commits)
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/facebookresearch/vggt.git@44b3afbd1869d8bde4894dd8ea1e293112dd5eba" \
    "git+https://github.com/facebookresearch/sam3.git@f6e51f59500a87c576c2df2323ce56b9fd7a12de"

# Pre-cache VGG-16 for LPIPS loss
RUN python3 -c "import torch; torch.hub.load_state_dict_from_url( \
    'https://download.pytorch.org/models/vgg16-397923af.pth', check_hash=True)"

# Copy only this application (context is HoloHub root)
COPY applications/surgical_scene_recon /workspace/app

WORKDIR /workspace
