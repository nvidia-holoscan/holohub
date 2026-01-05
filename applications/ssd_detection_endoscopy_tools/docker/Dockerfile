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
ARG ONNX_RUNTIME_VERSION=1.12.1
ARG BUILD_LIBTORCH=true
ARG LIBTORCH_VERSION=1.12.0
ARG TORCHVISION_VERSION=0.14.1
ARG VULKAN_SDK_VERSION=1.3.216.0
ARG GRPC_VERSION=1.54.2
ARG UCX_VERSION=1.14.0
ARG GXF_VERSION=23.05_20230717_d105fa1c

############################################################
# Base image
############################################################
ARG GPU_TYPE=dgpu
FROM nvcr.io/nvidia/pytorch:22.03-py3 AS dgpu_base
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel AS igpu_base
FROM ${GPU_TYPE}_base AS base

ARG DEBIAN_FRONTEND=noninteractive

# Remove any cuda list which could be invalid and prevent successful apt update
RUN rm -f /etc/apt/sources.list.d/cuda.list

############################################################
# Build tools
############################################################
FROM base AS build-tools

# Install build tools, including newer CMake (https://apt.kitware.com/)
RUN rm -rf \
    /usr/local/bin/cmake \
    /usr/local/bin/cpack \
    /usr/local/bin/ctest \
    /usr/local/share/cmake-3.14
RUN curl -s -L https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt update \
    && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
    && apt install --no-install-recommends -y \
        kitware-archive-keyring \
        cmake-data="3.22.2-*" \
        cmake="3.22.2-*" \
        patchelf \
        ninja-build="1.10.0-*" \
    && rm -rf /var/lib/apt/lists/*

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

############################################################
# NGC CLI
############################################################
FROM base AS ngc-cli-downloader

WORKDIR /opt/ngc-cli

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && curl -S -# -L -o ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && export ngc_exec=$(find . -type f -executable -name "ngc" | head -n1)

############################################################
# ONNX Runtime
############################################################
FROM base AS onnxruntime-downloader
ARG ONNX_RUNTIME_VERSION=1.12.1

WORKDIR /opt/onnxruntime

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=aarch64; else ARCH=x64-gpu; fi \
    && ONNX_RUNTIME_NAME="onnxruntime-linux-${ARCH}-${ONNX_RUNTIME_VERSION}" \
    && curl -S -# -O -L https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/${ONNX_RUNTIME_NAME}.tgz \
    && mkdir -p ${ONNX_RUNTIME_VERSION} \
    && tar -xzf ${ONNX_RUNTIME_NAME}.tgz -C ${ONNX_RUNTIME_VERSION} --strip-components 1

############################################################
# Libtorch (x86_64)
############################################################
FROM build-tools AS torch-downloader-amd64
ARG BUILD_LIBTORCH
ARG LIBTORCH_VERSION

# x86_64: Download libtorch binaries and remove packaged cuda dependencies
WORKDIR /opt/libtorch/${LIBTORCH_VERSION}
RUN if [ ${BUILD_LIBTORCH} = true ]; then \
        curl -S -# -o libtorch.zip -L \
            https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu116.zip \
        && unzip -q libtorch.zip \
        && rm libtorch.zip && mv libtorch/* . && rm -r libtorch \
        && rm lib/lib{cu,nv}*.so* \
        && regex_in="lib(cu|nv)(.*?)-[0-9a-f]{8}(.so[\.0-9]*)" \
        && regex_out="lib\\1\\2\\3" \
        && torch_libs=$(find $(pwd)/lib -type f -name "*.so") \
        && for torch_lib in $torch_libs; do \
            cuda_libs=$(ldd "$torch_lib" | grep -Eo "$regex_in" | sort -u) \
            && for cuda_lib in $cuda_libs; do \
                new_name=$(echo "$cuda_lib" | sed -E "s/$regex_in/$regex_out/") \
                && patchelf --replace-needed "$cuda_lib" "$new_name" "$torch_lib" \
                && echo "Updated $torch_lib: $cuda_lib -> $new_name"; \
            done; \
        done; \
    fi

############################################################
# Libtorch (aarch64)
############################################################
FROM base AS torch-downloader-arm64
ARG BUILD_LIBTORCH
ARG LIBTORCH_VERSION

# aarch64: Download libtorch binaries from artifactory
WORKDIR /opt/libtorch/${LIBTORCH_VERSION}
RUN if [ ${BUILD_LIBTORCH} = true ]; then \
        curl -S -# -o libtorch.zip -L \
            https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/libtorch/libtorch-${LIBTORCH_VERSION}-cuda-11.6-aarch64.zip \
        && unzip -q libtorch.zip \
        && rm libtorch.zip && mv libtorch/* . && rm -r libtorch; \
    fi

############################################################
# Libtorch
############################################################
FROM torch-downloader-${TARGETARCH} AS torch-downloader

############################################################
# TorchVision (x86_64)
############################################################
FROM torch-downloader-amd64 AS torchvision-builder-amd64
ARG BUILD_LIBTORCH
ARG TORCHVISION_VERSION
ARG LIBTORCH_VERSION

# x86_64: Build & install torchvision from source
WORKDIR /opt/torchvision/${TORCHVISION_VERSION}
RUN if [ ${BUILD_LIBTORCH} = true ]; then \
        curl -S -# -o torchvision.zip -L \
            https://github.com/pytorch/vision/archive/refs/tags/v${TORCHVISION_VERSION}.zip \
        && unzip -q torchvision.zip \
        && rm torchvision.zip && mv vision-${TORCHVISION_VERSION} src \
        && cmake -S src -B build -G Ninja \
            -DWITH_CUDA=on -DWITH_PNG=off -DWITH_JPEG=off \
            -DTorch_ROOT=/opt/libtorch/${LIBTORCH_VERSION} \
        && cmake --build build -j $(nproc) \
        && cmake --install build --prefix $(pwd) \
        && rm -r src build; \
    fi

############################################################
# TorchVision (aarch64)
############################################################
FROM base AS torchvision-builder-arm64
ARG BUILD_LIBTORCH
ARG TORCHVISION_VERSION

# aarch64: Download torchvision binaries from artifactory
WORKDIR /opt/torchvision/${TORCHVISION_VERSION}
RUN if [ ${BUILD_LIBTORCH} = true ]; then \
        curl -S -# -o torchvision.zip -L \
            https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/torchvision/torchvision-${TORCHVISION_VERSION}-cuda-11.6-aarch64.zip \
        && unzip -q torchvision.zip \
        && rm torchvision.zip && mv torchvision/* . && rm -r torchvision; \
    fi

############################################################
# TorchVision
############################################################
FROM torchvision-builder-${TARGETARCH} AS torchvision-builder

############################################################
# Vulkan SDK
#
# Use the SDK because we need the newer Vulkan headers and the newer shader compiler than provided
# by the Ubuntu deb packages. These are compile time dependencies, we still use the Vulkan loaded
# and the Vulkan validation layer as runtime components provided by Ubuntu packages because that's
# what the user will have on their installations.
############################################################
FROM build-tools AS vulkansdk-builder
ARG VULKAN_SDK_VERSION

WORKDIR /opt/vulkansdk

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and e7ab9314build the aarch64 version from source.
RUN curl -S -# -O -L https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) = "aarch64" ]; then \
    cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && ./vulkansdk shaderc glslang headers; \
    fi

############################################################
# gRPC libraries and binaries
############################################################
FROM build-tools AS grpc-builder
ARG GRPC_VERSION

WORKDIR /opt/grpc
RUN git clone --depth 1 --branch v${GRPC_VERSION} \
         --recurse-submodules --shallow-submodules \
         https://github.com/grpc/grpc.git src
RUN cmake -S src -B build -G Ninja \
         -D CMAKE_BUILD_TYPE=Release \
         -D gRPC_INSTALL=ON \
         -D gRPC_BUILD_TESTS=OFF
RUN cmake --build build -j $(nproc)
RUN cmake --install build --prefix /opt/grpc/${GRPC_VERSION}

############################################################
# UCX
############################################################
FROM build-tools AS ucx-builder
ARG UCX_VERSION

# Clone
WORKDIR /opt/ucx/
RUN git clone --depth 1 --branch v${UCX_VERSION} https://github.com/openucx/ucx.git src

# Apply patches for iGPU compatibility
WORKDIR /opt/ucx/src
COPY patches/ucx_*.patch ..
RUN git apply ../ucx*.patch

# Prerequisites to build
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libtool="2.4.6-*" \
        automake="1:1.16.1-*" \
    && rm -rf /var/lib/apt/lists/*

# Build and install
RUN ./autogen.sh
WORKDIR /opt/ucx/build
RUN ../src/contrib/configure-release --enable-mt --with-cuda=/usr/local/cuda-11 \
    --prefix=/opt/ucx/${UCX_VERSION}
RUN make -j install

# Apply patches for import and run
WORKDIR /opt/ucx/${UCX_VERSION}
# patch cmake config
RUN sed -i "s|set(prefix.*)|set(prefix \"$(pwd)\")|" lib/cmake/ucx/ucx-targets.cmake
# patch rpath (relative to ORIGIN)
RUN patchelf --set-rpath '$ORIGIN' lib/libuc*.so*
RUN patchelf --set-rpath '$ORIGIN:$ORIGIN/..' lib/ucx/libuc*.so*
RUN patchelf --set-rpath '$ORIGIN/../lib' bin/*

############################################################
# GXF
############################################################
FROM base AS gxf-downloader
ARG GXF_VERSION

WORKDIR /opt/nvidia/gxf
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=x86_64; fi \
    && curl -S -# -L -o gxf.tgz \
        https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/gxf/gxf_${GXF_VERSION}_holoscan-sdk_${ARCH}.tar.gz
RUN mkdir -p ${GXF_VERSION}
RUN tar -xzf gxf.tgz -C ${GXF_VERSION} --strip-components 1

# Patch GXF to remove its support for complex cuda primitives,
# due to limitation in the CUDA Toolkit 11.6 with C++17
# Support will be added when upgrading to CUDA Toolkit 11.7+
# with libcu++ 1.8.0+: https://github.com/NVIDIA/libcudacxx/pull/234
COPY patches/gxf_remove_complex_primitives_support.patch .
RUN patch -suNb ${GXF_VERSION}/gxf/std/tensor.hpp gxf_remove_complex_primitives_support.patch
RUN mv ${GXF_VERSION}/gxf/std/complex.hpp ${GXF_VERSION}/gxf/std/complex.hpp.orig

############################################################
# dev image
# Dev (final)
############################################################
FROM build-tools AS dev

## INSTALLS

# Install apt & pip tools
RUN apt update \
    && apt install --no-install-recommends -y \
        lcov="1.14-*" \
        valgrind="1:3.15.0-*" \
    && rm -rf /var/lib/apt/lists/*

COPY python/requirements.dev.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.dev.txt

# Install apt & pip build dependencies
#  libvulkan1 - for Vulkan apps (Holoviz)
#  vulkan-validationlayers, spirv-tools - for Vulkan validation layer (enabled for Holoviz in debug mode)
#  libegl1 - to run headless Vulkan apps
#  libopenblas0 - libtorch dependency
#  libv4l-dev - V4L2 operator dependency
#  v4l-utils - V4L2 operator utility
RUN apt update \
    && apt install --no-install-recommends -y \
        libgl-dev="1.3.2-*" \
        libx11-dev="2:1.6.9-*" \
        libxcursor-dev="1:1.2.0-*" \
        libxext-dev="2:1.3.4-*" \
        libxi-dev="2:1.7.10-*" \
        libxinerama-dev="2:1.1.4-*" \
        libxrandr-dev="2:1.5.2-*" \
        libxxf86vm-dev="1:1.1.4-*" \
        libvulkan1="1.2.131.2-*" \
        vulkan-validationlayers="1.2.131.2-*" \
        spirv-tools="2020.1-*" \
        libegl1="1.3.2-*" \
        libopenblas0="0.3.8+ds-*" \
        libv4l-dev="1.18.0-*" \
        v4l-utils="1.18.0-*" \
    && rm -rf /var/lib/apt/lists/*

# Install pip run dependencies
RUN if [ $(uname -m) = "aarch64" ]; then \
        python3 -m pip install cupy-cuda11x~=11.3 -f https://pip.cupy.dev/aarch64; \
    else \
        python3 -m pip install cupy-cuda11x~=11.3; \
    fi
# Install NVTC
RUN python3 -m pip install nvtx

# Install pip dependencies for CLI
COPY python/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install ffmpeg
RUN ARCH=$(uname -m) \
    && curl -S -# -o /usr/bin/ffmpeg \
             -L https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/ffmpeg/6.0/bin/${ARCH}/ffmpeg \
    && chmod +x /usr/bin/ffmpeg

## PATCHES

# Setup EGL for running headless Vulkan apps
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo -e "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\" : {\n\
        \"library_path\" : \"libEGL_nvidia.so.0\"\n\
    }\n\
}\n" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install symlink missing from the container for H264 encode examples
RUN if [ $(uname -m) = "x86_64" ]; then \
        ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so; \
    fi

## COPY
# Note: avoid LD_LIBRARY_PATH - https://www.hpc.dtu.dk/?page_id=1180

# Copy NGC CLI
ENV NGC_CLI=/opt/ngc-cli
COPY --from=ngc-cli-downloader ${NGC_CLI}/ngc-cli ${NGC_CLI}
ENV PATH="${PATH}:${NGC_CLI}"
# Workaround to fix encoding for ngc-cli download
# https://jirasw.nvidia.com/browse/NGC-31306
ENV PYTHONIOENCODING=utf-8 LC_ALL=C.UTF-8

# Copy ONNX Runtime
ARG ONNX_RUNTIME_VERSION
ENV ONNX_RUNTIME=/opt/onnxruntime/${ONNX_RUNTIME_VERSION}
COPY --from=onnxruntime-downloader ${ONNX_RUNTIME} ${ONNX_RUNTIME}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${ONNX_RUNTIME}"

# Copy Libtorch
ARG LIBTORCH_VERSION
ENV LIBTORCH=/opt/libtorch/${LIBTORCH_VERSION}
COPY --from=torch-downloader ${LIBTORCH} ${LIBTORCH}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${LIBTORCH}"

# Copy TorchVision
ARG TORCHVISION_VERSION
ENV TORCHVISION=/opt/torchvision/${TORCHVISION_VERSION}
COPY --from=torchvision-builder ${TORCHVISION} ${TORCHVISION}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${TORCHVISION}"

# Copy Vulkan SDK
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

# Copy gRPC
ARG GRPC_VERSION
ENV GRPC=/opt/grpc/${GRPC_VERSION}
COPY --from=grpc-builder ${GRPC} ${GRPC}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GRPC}"

# Copy UCX
ARG UCX_VERSION
ENV UCX=/opt/ucx/${UCX_VERSION}
COPY --from=ucx-builder ${UCX} ${UCX}
ENV PATH="${PATH}:${UCX}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${UCX}"
# remove older version of UCX in hpcx install
RUN rm -rf /opt/hpcx/ucx /usr/local/ucx
RUN unset OPENUCX_VERSION
# required for gxf_ucx.so to find ucx
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${UCX}/lib"

# Copy GXF
ARG GXF_VERSION
ENV GXF=/opt/nvidia/gxf/${GXF_VERSION}
COPY --from=gxf-downloader ${GXF} ${GXF}
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${GXF}"
