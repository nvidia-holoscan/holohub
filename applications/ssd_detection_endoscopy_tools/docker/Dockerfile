# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Versions
############################################################
ARG TRT_CONTAINER_TAG=22.03-py3
ARG ONNX_RUNTIME_VERSION=1.12.1
ARG VULKAN_SDK_VERSION=1.3.216.0

############################################################
# Base image
############################################################
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${TRT_CONTAINER_TAG}
FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# Install newer CMake (https://apt.kitware.com/)
RUN rm -rf \
    /usr/local/bin/cmake \
    /usr/local/bin/cpack \
    /usr/local/bin/ctest \
    /usr/local/share/cmake-3.14
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt update \
    && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
    && apt install --no-install-recommends -y \
        kitware-archive-keyring \
        cmake-data=3.22.2-0kitware1ubuntu20.04.1 \
        cmake=3.22.2-0kitware1ubuntu20.04.1 \
    && rm -rf /var/lib/apt/lists/*

# Install symlink missing from the container for H264 encode examples
RUN if [ $(uname -m) == "x86_64" ]; then \
        ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so; \
    fi

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

############################################################
# NGC CLI
############################################################
FROM base as ngc-cli-downloader

WORKDIR /opt/ngc-cli

RUN if [ $(uname -m) == "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && wget \
        -nv --show-progress --progress=bar:force:noscroll \
        -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -o ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1)

############################################################
# ONNYX Runtime
############################################################
FROM base as onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION=1.12.1

WORKDIR /opt/onnxruntime

RUN if [ $(uname -m) == "aarch64" ]; then ARCH=aarch64; else ARCH=x64-gpu; fi \
    && ONNX_RUNTIME_NAME="onnxruntime-linux-${ARCH}-${ONNX_RUNTIME_VERSION}" \
    && wget \
        -nv --show-progress --progress=bar:force:noscroll \
        https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/${ONNX_RUNTIME_NAME}.tgz \
    && mkdir -p ${ONNX_RUNTIME_VERSION} \
    && tar -xzf ${ONNX_RUNTIME_NAME}.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 1

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

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and build the aarch64 version from source.
RUN wget -nv --show-progress --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) == "aarch64" ]; then \
    cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && ./vulkansdk shaderc glslang headers; \
    fi


############################################################
# dev image
############################################################
FROM base as dev

# Install apt & pip tools
RUN apt update \
    && apt install --no-install-recommends -y \
        patchelf \
        lcov=1.14-2 \
        python3-pytest \
        ninja-build=1.10.0-1build1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install \
        coverage==6.5.0

# Install apt & pip build dependencies
#  libvulkan1 - for Vulkan apps (Holoviz)
#  vulkan-validationlayers, spirv-tools - for Vulkan validation layer (enabled for Holoviz in debug mode)
#  libegl1 - to run headless Vulkan apps
RUN apt update \
    && apt install --no-install-recommends -y \
        libgl-dev=1.3.2-1~ubuntu0.20.04.2 \
        libx11-dev=2:1.6.9-2ubuntu1.2 \
        libxcursor-dev=1:1.2.0-2 \
        libxext-dev=2:1.3.4-0ubuntu1 \
        libxi-dev=2:1.7.10-0ubuntu1 \
        libxinerama-dev=2:1.1.4-2 \
        libxrandr-dev=2:1.5.2-0ubuntu1 \
        libxxf86vm-dev=1:1.1.4-1build1 \
        libvulkan1=1.2.131.2-1 \
        vulkan-validationlayers=1.2.131.2-1 \
        spirv-tools=2020.1-2 \
        libegl1=1.3.2-1~ubuntu0.20.04.2 \
    && rm -rf /var/lib/apt/lists/*

# Install pip run dependencies
RUN if [ $(uname -m) == "aarch64" ]; then \
        python3 -m pip install cupy-cuda11x==11.3.0 -f https://pip.cupy.dev/aarch64; \
    else \
        python3 -m pip install cupy-cuda11x==11.3.0; \
    fi
#install NVTX for profiling
RUN python3 -m pip install nvtx


## COPY BUILT/DOWNLOADED dependencies in previous stages
# Note: avoid LD_LIBRARY_PATH - https://www.hpc.dtu.dk/?page_id=1180

# Copy NGC CLI
ENV NGC_CLI=/opt/ngc-cli
COPY --from=ngc-cli-downloader ${NGC_CLI}/ngc-cli ${NGC_CLI}
ENV PATH="${PATH}:${NGC_CLI}"

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}
COPY --from=onnxruntime-downloader ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Vulkan SDK
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
COPY --from=vulkansdk-builder ${VULKAN_SDK}/x86_64/ ${VULKAN_SDK}
# We need to use the headers and shader compiler of the SDK but want to link against the
# Vulkan loader provided by the Ubuntu package. Therefore create a link in the SDK directory
# pointing to the system Vulkan loader library.
RUN rm -f ${VULKAN_SDK}/lib/libvulkan.so* \
    && ln -s /lib/$(uname -m)-linux-gnu/libvulkan.so.1 ${VULKAN_SDK}/lib/libvulkan.so
# Setup EGL for running headless Vulkan apps
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo -e "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\" : {\n\
        \"library_path\" : \"libEGL_nvidia.so.0\"\n\
    }\n\
}\n" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV PATH="${PATH}:${VULKAN_SDK}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VULKAN_SDK}"
