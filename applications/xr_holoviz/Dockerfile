# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install Vulkan SDK and ICD.
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
RUN apt update \
    && apt install --no-install-recommends -y \
        vulkan-icd \
        vulkan-sdk

# Install OpenXR SDK Dependencies.
RUN apt update \
    && apt install --no-install-recommends -y \
        build-essential \
        cmake \
        libeigen3-dev \
        libgl1-mesa-dev \
        libvulkan-dev \
        libwayland-dev\
        libx11-xcb-dev \
        libxcb-dri2-0-dev \
        libxcb-glx0-dev \
        libxcb-icccm4-dev \
        libxcb-keysyms1-dev \
        libxcb-randr0-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        mesa-common-dev \
        libopenxr-loader1 \
        libopenxr-dev \
        libopenxr1-monado

# Install Python builder.
RUN pip install --upgrade pip
RUN pip install build

# Set the working directory
WORKDIR /app/xr_holoviz

#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*
