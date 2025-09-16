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

############################################################
# Base image
############################################################
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
FROM ${BASE_IMAGE}

# Install system dependencies including updated OpenSSL and testing tools
RUN apt-get update && apt-get install -y \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    gir1.2-girepository-2.0 \
    gobject-introspection \
    python3-gi \
    libopencv-dev \
    ffmpeg \
    wget \
    unzip \
    software-properties-common \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies for OpenSSL
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install OpenSSL 3.4.0 to match bundled library requirements
RUN cd /tmp && \
    wget https://github.com/openssl/openssl/releases/download/openssl-3.4.0/openssl-3.4.0.tar.gz && \
    tar -xzf openssl-3.4.0.tar.gz && \
    cd openssl-3.4.0 && \
    ./Configure --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib && \
    make -j$(nproc) && \
    make install && \
    echo "/usr/local/ssl/lib64" > /etc/ld.so.conf.d/openssl.conf && \
    ldconfig && \
    cd / && rm -rf /tmp/openssl-3.4.0*

# Set library path for OpenSSL
ENV LD_LIBRARY_PATH="/usr/local/ssl/lib64:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="/usr/local/ssl/lib64/pkgconfig:${PKG_CONFIG_PATH}"

# Setup holohub run environment
ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib:${PYTHONPATH}

WORKDIR /workspace/holohub

# Create workspace directories
RUN mkdir -p /workspace/holohub/applications/streaming_server_demo_04_80_tensor_streaming

# Copy the rest of the application files
COPY . /workspace/holohub/

# Create data directory
RUN mkdir -p /workspace/data/endoscopy

# Set working directory
WORKDIR /workspace/holohub

# Set environment variables
ENV HOLOSCAN_INPUT_PATH=/workspace/data/endoscopy
ENV PYTHONPATH=/workspace/holohub/applications/streaming_server_demo/python:${PYTHONPATH}

# Default command
CMD ["/bin/bash"]

