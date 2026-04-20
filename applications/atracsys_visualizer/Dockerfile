# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.0.0-cuda12-dgpu
ARG GPU_TYPE=dgpu
ARG OPENCV_VERSION=4.10.0
ARG CUDA_ARCH_BIN=""
ARG CUDA_ARCH_PTX=""

FROM ${BASE_IMAGE} AS opencv_builder

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_PTX

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      cmake \
      git \
      gfortran \
      libatlas-base-dev \
      libavcodec-dev \
      libavformat-dev \
      libjpeg-dev \
      libpng-dev \
      libswscale-dev \
      libtbb-dev \
      libtiff-dev \
      libv4l-dev \
      libx264-dev \
      libxvidcore-dev \
      python3-dev \
      unzip \
      wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

RUN mkdir -p /opt/opencv/build && \
    cd /opt/opencv/build && \
    OPENCV_CUDA_ARGS="" && \
    if [ -n "${CUDA_ARCH_BIN}" ]; then OPENCV_CUDA_ARGS="${OPENCV_CUDA_ARGS} -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN}"; fi && \
    if [ -n "${CUDA_ARCH_PTX}" ]; then OPENCV_CUDA_ARGS="${OPENCV_CUDA_ARGS} -D CUDA_ARCH_PTX=${CUDA_ARCH_PTX}"; fi && \
    cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_STANDARD_REQUIRED=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D BUILD_TBB=OFF \
      -D BUILD_opencv_world=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_JPEG=ON \
      -D BUILD_PNG=ON \
      -D BUILD_TIFF=ON \
      -D BUILD_WEBP=ON \
      -D BUILD_ZLIB=ON \
      -D WITH_FFMPEG=OFF \
      -D WITH_GSTREAMER=OFF \
      -D CUDA_NVCC_FLAGS=--std=c++17 \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
      ${OPENCV_CUDA_ARGS} \
      .. && \
    cmake --build . --parallel "$(nproc)" && \
    cmake --install .

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

COPY --from=opencv_builder /usr/local /usr/local

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      build-essential \
      cmake \
      git \
      libtbb-dev \
      libvulkan1 \
      libyaml-cpp-dev \
      unzip \
      vulkan-tools && \
    rm -rf /var/lib/apt/lists/*

# Fail the build if the required custom OpenCV 4.10 is missing, preventing silent S3DK crashes.
# Keep S3DK-compatible sonames available (aliasing to .410 since S3DK is tested on OpenCV 4.10)
RUN if [ ! -f /usr/local/lib/libopencv_world.so.4.10.0 ]; then \
      echo "ERROR: OpenCV 4.10.0 artifact not found in /usr/local/lib/" && exit 1; \
    fi && \
    ln -sf /usr/local/lib/libopencv_world.so.4.10.0 /usr/local/lib/libopencv_world.so.410 && \
    ln -sf /usr/local/lib/libopencv_world.so.4.10.0 /usr/local/lib/libopencv_world.so.4.10 && \
    ldconfig

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
ENV OpenCV_DIR=/usr/local/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
