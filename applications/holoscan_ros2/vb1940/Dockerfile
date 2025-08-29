# syntax=docker/dockerfile:1

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
# Build the vb1940 image with ROS2 and hololink dependencies
############################################################
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
FROM ${BASE_IMAGE} AS vb1940

# Install basic dependencies
RUN apt update && apt install -y \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install git and openssh-client so we can clone the hololink repository
RUN apt update && apt install -y \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Jazzy repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy and comprehensive Vulkan support
RUN apt update && apt install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-base \
    vulkan-tools \
    vulkan-validationlayers \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Set ROS2 environment variables directly
ENV ROS_DISTRO=jazzy
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV CMAKE_PREFIX_PATH=/opt/ros/jazzy:/opt/nvidia/holoscan
ENV AMENT_PREFIX_PATH=/opt/ros/jazzy
ENV LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/ros/jazzy/bin:$PATH
ENV PKG_CONFIG_PATH=/opt/ros/jazzy/lib/pkgconfig

# Set comprehensive Vulkan environment variables for Holoviz
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d
ENV VK_LOADER_LAYERS_ENABLE=1
ENV VK_LOADER_DEBUG=all

# Set PYTHONPATH to include ROS2 packages
ENV PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:/opt/nvidia/holoscan/python/lib

# Create necessary directories for Vulkan
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/vulkan/explicit_layer.d

# Create a proper NVIDIA Vulkan ICD configuration
#RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.242"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN echo -e '#!/bin/bash \n \
set -e \n \
source /opt/ros/jazzy/setup.bash \n \
exec "$@"' > /usr/bin/entrypoint.sh && chmod +x /usr/bin/entrypoint.sh

ENTRYPOINT ["/usr/bin/entrypoint.sh"]

# Set up deps directory and clone holoscan-sensor-bridge
RUN mkdir -p /deps

# Add SSH key and known hosts during build
RUN mkdir -p /root/.ssh && \
    ssh-keyscan -p 12051 gitlab-master.nvidia.com >> /root/.ssh/known_hosts

# Clone the Hololink repository
WORKDIR /deps
RUN --mount=type=ssh git clone --branch internal ssh://git@gitlab-master.nvidia.com:12051/holoscan/hololink.git holoscan-sensor-bridge

# Build the hololink C++ libraries
WORKDIR /deps/holoscan-sensor-bridge
RUN mkdir -p build
WORKDIR /deps/holoscan-sensor-bridge/build
RUN cmake .. && make -j$(nproc)

# Install the hololink Python package
WORKDIR /deps/holoscan-sensor-bridge/python
RUN pip3 install -e .

# Set default working directory
WORKDIR /workspace
