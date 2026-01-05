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
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub dev setup
#

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Install libraries
RUN apt update && \
    apt install -y --no-install-recommends \
    libboost-all-dev \
    libgtk2.0-dev \
    portaudio19-dev

# Build OpenCV 4.5.5 (needed by ProHawk SDK)
RUN if [ $(uname -m) = "aarch64" ]; then CMAKE_ARGS="-DENABLE_NEON=ON"; \
    else CMAKE_ARGS="-DENABLE_NEON=OFF"; fi \
    && git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib \
    && git checkout 4.5.5 && cd .. \
 	 && cd opencv && git checkout 4.5.5 \
    && mkdir -p build && cd build && \
  	 cmake -DCMAKE_BUILD_TYPE=RELEASE \
	 -DCMAKE_INSTALL_PREFIX=/usr/local \
	 -DBUILD_PNG=OFF \
     -DBUILD_TIFF=OFF \
     -DBUILD_TBB=OFF \
     -DBUILD_JPEG=OFF \
     -DBUILD_JASPER=OFF \
     -DBUILD_ZLIB=OFF \
     -DBUILD_EXAMPLES=OFF \
     -DBUILD_JAVA=OFF \
     -DBUILD_opencv_python2=OFF \
     -DBUILD_opencv_python3=OFF \
     -DBUILD_TESTS=OFF \
     -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
     -DWITH_OPENCL=OFF \
     -DWITH_OPENMP=OFF \
     -DWITH_FFMPEG=OFF \
     -DWITH_GSTREAMER=OFF \
     -DWITH_GSTREAMER_0_10=OFF \
     -DWITH_CUDA=ON \
     -DWITH_GTK=ON \
     -DWITH_VTK=OFF \
     -DWITH_TBB=ON \
     -DWITH_1394=OFF \
     -DWITH_OPENEXR=OFF \
     -DCUDA_ARCH_BIN=8.6 \
     -DINSTALL_C_EXAMPLES=OFF \
     -DINSTALL_TESTS=OFF \
     -DBUILD_EXAMPLES=OFF \
     ${CMAKE_ARGS} \
	 .. && \
	make -j8 && \
	make install && \
	ldconfig

# Install prohawk runtime
RUN wget https://prohawkholoscanv051runtime.s3.us-west-2.amazonaws.com/phruntime_v60_aarch64.zip && unzip phruntime_v60_aarch64.zip -d /usr/local

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# Install Riva client and pyaudio to enable Python ASR dev
RUN pip install \
    nvidia-riva-client \
    pyaudio

# Copy the filters to be in the same directory as the Python executable so the application
# can find them at runtime
COPY applications/prohawk_video_replayer/cpp/PAFSFilter.bin /usr/bin
