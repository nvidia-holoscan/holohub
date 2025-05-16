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

ARG DEBIAN_FRONTEND=noninteractive

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu as base


############################################################
# HoloHub run + dev dependencies setup
############################################################

RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

############################################################
# Magic Leap base setup
############################################################

COPY /applications/volume_rendering_xr/scripts/base-setup.sh \
    /tmp/base-setup.sh
COPY applications/volume_rendering_xr/thirdparty/magicleap/MagicLeapRemoteRendering.gpg \
    /usr/local/share/keyrings/magicleap/MagicLeapRemoteRendering.gpg
RUN /tmp/base-setup.sh

############################################################
# OpenXR dev dependencies
############################################################
FROM base as dev

ARG DEBIAN_FRONTEND=noninteractive

# Install OpenXR dev dependencies, can remove when using libopenxr-dev package instead of building OpenXR from source
RUN apt update \
    && apt install --no-install-recommends -y \
        libgl1-mesa-dev \
        libx11-xcb-dev \
        libxcb-dri2-0-dev \
        libxcb-glx0-dev \
        libxcb-icccm4-dev \
        libxcb-keysyms1-dev \
        libxcb-randr0-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        mesa-common-dev \
        libvulkan-dev="1.3.204.1-*" \
    && rm -rf /var/lib/apt/lists/*

# Install specific version of markupsafe for OpenXR build
RUN pip install markupsafe==2.0.1

############################################################
# Windrunner configuration
# 
# Add known IGX dGPU syscall requirements to the Windrunner
# seccomp whitelist
############################################################
COPY /applications/volume_rendering_xr/thirdparty/magicleap/seccomp_whitelist.cfg \
     /usr/share/windrunner/seccomp_whitelist.cfg

############################################################
# Environment variables
############################################################
# Magic Leap Windrunner environment variables
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/windrunner/lib"
ENV LR_TRANSPORT_CERT_PATH=/opt/windrunner/share/windrunner/server.crt
ENV LR_TRANSPORT_PKEY_PATH=/opt/windrunner/share/windrunner/server.key
ENV WINDRUNNER_ENABLE_AUDIO=0
ENV PATH="${PATH}:/opt/windrunner/bin"

# Setup Magic Leap CLI
ENV PATH="${PATH}:/workspace/holohub/applications/volume_rendering_xr/thirdparty/magicleap"

# Set JIT compilation cache paths
#  https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
ENV CUDA_CACHE_DISABLE="0"
ENV CUDA_CACHE_PATH="/workspace/holohub/.cache/ComputeCache"
#  https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#a59a60f5f600df0f9321b0a0b1090d76b
ENV OPTIX_CACHE_PATH="/workspace/holohub/.cache/OptixCache"
#  https://download.nvidia.com/XFree86/Linux-x86_64/460.67/README/openglenvvariables.html
ENV __GL_SHADER_DISK_CACHE_PATH="/workspace/holohub/.cache/GLCache"
