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
#
# Build context: HoloHub repo root (same as surgical_scene_recon).
# Base image: NGC Holoscan (passed by holohub build-container).
# No local base image; no COPY . from app dir.
############################################################

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
# Set up prerequisites to run HoloHub CLI (same as surgical_scene_recon)
# --------------------------------------------------------------------------
FROM base AS holohub-cli-prerequisites

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
# HoloHub CLI setup (same as surgical_scene_recon)
# --------------------------------------------------------------------------
FROM holohub-cli-prerequisites AS holohub-cli

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

# --------------------------------------------------------------------------
# G-SHARP application dependencies and app copy
# Build context is HoloHub root; copy only this application.
# --------------------------------------------------------------------------
FROM holohub-cli AS application-dependencies

COPY applications/gsplat_scene_recon/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Fix decord↔libxcb conflict for HoloViz/Vulkan
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libxcb1 libxcb-xfixes0 libxcb-shape0 libxcb-shm0 libxcb-render0 \
        libxcb-xinerama0 libxcb-randr0 && \
    rm -rf /var/lib/apt/lists/* && \
    DECORD_LIBS=$(python3 -c "import pathlib; print(next(pathlib.Path('/usr/local/lib').rglob('decord.libs'), ''))" 2>/dev/null) && \
    if [ -n "$DECORD_LIBS" ] && [ -d "$DECORD_LIBS" ]; then \
        for f in "$DECORD_LIBS"/libxcb*.so*; do \
            [ -e "$f" ] || continue; \
            base=$(basename "$f" | sed 's/-[a-f0-9]*\./\./') && \
            sys_lib=$(find /usr/lib/$(dpkg-architecture -q DEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu) -name "${base%%.*}*" -type f 2>/dev/null | head -1) && \
            if [ -n "$sys_lib" ]; then ln -sf "$sys_lib" "$f"; fi; \
        done; \
    fi

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
COPY applications/gsplat_scene_recon /workspace/app

WORKDIR /workspace
