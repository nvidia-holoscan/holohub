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
ARG VULKAN_SDK_VERSION=1.3.216.0

ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

COPY /applications/volume_rendering_xr/scripts/base-setup.sh \
    /tmp/base-setup.sh
COPY applications/volume_rendering_xr/thirdparty/magicleap/MagicLeapRemoteRendering.gpg \
    /usr/local/share/keyrings/magicleap/MagicLeapRemoteRendering.gpg
RUN /tmp/base-setup.sh

############################################################
# Vulkan SDK
#
# Use the SDK because we need the newer Vulkan headers and the newer shader compiler than provided
# by the Ubuntu deb packages. These are compile time dependencies, we still use the Vulkan loaded
# and the Vulkan validation layer as runtime components provided by Ubuntu packages because that's
# what the user will have on their installations.
############################################################
FROM base as vulkansdk-builder
ARG VULKAN_SDK_VERSION

WORKDIR /opt/vulkansdk
ENV CMAKE_GENERATOR=Ninja

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and build the aarch64 version from source.
RUN wget -nv --show-progress --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) = "aarch64" ]; then \
    apt update \
    && apt install --no-install-recommends -y python-is-python3 python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && MAKEFLAGS="-j $(nproc)" \
    && unset CMAKE_GENERATOR \
    && ./vulkansdk shaderc glslang headers; \
    fi

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
    && rm -rf /var/lib/apt/lists/*

# Copy vulkan sdk
# NOTE: It's all in x86_64 even if that's not the target platform
# (Vulkan SDK cmake scripting issue)
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
COPY --from=vulkansdk-builder ${VULKAN_SDK}/x86_64/ ${VULKAN_SDK}

# We need to use the headers and shader compiler of the SDK but want to link against the
# Vulkan loader provided by the Ubuntu package. Therefore create a link in the SDK directory
# pointing to the system Vulkan loader library.
RUN rm -f ${VULKAN_SDK}/lib/libvulkan.so* \
    && ln -s /lib/$(uname -m)-linux-gnu/libvulkan.so.1 ${VULKAN_SDK}/lib/libvulkan.so
ENV PATH="${PATH}:${VULKAN_SDK}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VULKAN_SDK}"

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
