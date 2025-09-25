# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu
FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    gir1.2-girepository-2.0 \
    gobject-introspection \
    python3-gi \
    ffmpeg \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for testing
RUN python3 -m pip install --no-cache-dir pytest

# Setup holohub run environment
ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib:${PYTHONPATH}

WORKDIR /workspace/holohub

# Create workspace directories
RUN mkdir -p /workspace/holohub/applications/streaming_server_demo

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

