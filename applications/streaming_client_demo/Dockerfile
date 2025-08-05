# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
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
    libopencv-dev \
    ffmpeg \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install NGC CLI following official instructions
WORKDIR /tmp/ngc-install
RUN wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.151.0/files/ngccli_linux.zip -O ngccli_linux.zip && \
    unzip ngccli_linux.zip && \
    # Verify SHA256
    echo "834aeebcdfee2f6db61744dbae5f976c4e3a7a4a9aa99d23dd1ed2928a2abd60 ngccli_linux.zip" | sha256sum --check && \
    # Make NGC CLI executable and add to path
    chmod u+x ngc-cli/ngc && \
    # Install to system path
    mkdir -p /usr/local/ngc && \
    mv ngc-cli /usr/local/ngc/ && \
    ln -s /usr/local/ngc/ngc-cli/ngc /usr/local/bin/ngc && \
    # Cleanup
    rm ngccli_linux.zip

# Add NGC to PATH
ENV PATH="/usr/local/ngc/ngc-cli:${PATH}"

# Setup holohub run environment
ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib:${PYTHONPATH}

WORKDIR /workspace/holohub

# Enable autocomplete
RUN echo "source /etc/bash_completion" >> ~/.bashrc

# Set CMake build type
ENV CMAKE_BUILD_TYPE=RelWithDebInfo

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# Create workspace directories
RUN mkdir -p /workspace/holohub/applications/streaming_client_demo

# Copy the rest of the application files
COPY . /workspace/holohub/

# Create data directory
RUN mkdir -p /workspace/data/endoscopy

# Set working directory
WORKDIR /workspace/holohub

# Set environment variables
ENV HOLOSCAN_INPUT_PATH=/workspace/data/endoscopy
ENV PYTHONPATH=/workspace/holohub/applications/streaming_client_demo/python:${PYTHONPATH}

# Default command
CMD ["/bin/bash"]

# Enable autocomplete
RUN echo "source /etc/bash_completion" >> ~/.bashrc

# Set CMake build type
ENV CMAKE_BUILD_TYPE=RelWithDebInfo

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# Install system dependencies FIRST
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 && \
    for i in {1..3}; do apt-get update && break || sleep 1; done && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    gir1.2-girepository-2.0 \
    gobject-introspection \
    python3-gi \
    libopencv-dev \
    libopencv-contrib-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to match StreamingClient expectations
# The system OpenCV will be installed, we just need to create the 4.5d links
RUN cd /usr/lib/x86_64-linux-gnu && \
    # Find the actual OpenCV version installed and create 4.5d symbolic links
    OPENCV_VERSION=$(ls libopencv_core.so.* | head -1 | sed 's/libopencv_core.so.//') && \
    echo "Found OpenCV version: $OPENCV_VERSION" && \
    ln -sf libopencv_core.so.$OPENCV_VERSION libopencv_core.so.4.5d && \
    ln -sf libopencv_imgproc.so.$OPENCV_VERSION libopencv_imgproc.so.4.5d && \
    ln -sf libopencv_highgui.so.$OPENCV_VERSION libopencv_highgui.so.4.5d && \
    ln -sf libopencv_videoio.so.$OPENCV_VERSION libopencv_videoio.so.4.5d && \
    ln -sf libopencv_imgcodecs.so.$OPENCV_VERSION libopencv_imgcodecs.so.4.5d && \
    ln -sf libopencv_dnn.so.$OPENCV_VERSION libopencv_dnn.so.4.5d || \
    echo "Some OpenCV libraries may not exist, continuing..."

# Then install Python packages
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Setup holohub run environment - updated to ensure Holoscan Python paths are properly set
ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib:/opt/nvidia/holoscan/lib:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/nvidia/holoscan/lib:${LD_LIBRARY_PATH}

# Create workspace directories
RUN mkdir -p /workspace/holohub/applications/streaming_client_demo

# Copy the rest of the application files
COPY . /workspace/holohub/

# Create data directory
RUN mkdir -p /workspace/data/endoscopy

# Set working directory
WORKDIR /workspace/holohub

# Set environment variables
ENV HOLOSCAN_INPUT_PATH=/workspace/data/endoscopy
ENV PYTHONPATH=/workspace/holohub/applications/streaming_client_demo/python:${PYTHONPATH}

# Set CMake build type
ENV CMAKE_BUILD_TYPE=RelWithDebInfo
ARG CMAKE_BUILD_TYPE=Release

# Enable autocomplete
RUN echo "source /etc/bash_completion" >> ~/.bashrc

# Default command
ENTRYPOINT ["/bin/bash"]
CMD ["/bin/bash"]

