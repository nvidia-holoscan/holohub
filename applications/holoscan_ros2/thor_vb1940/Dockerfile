# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Build the thor_vb1940 image with ROS2 and hololink dependencies
############################################################
ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS thor_vb1940

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# Install basic dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install git to clone the public holoscan-sensor-bridge repository
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Jazzy repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy and comprehensive Vulkan support
RUN apt-get update && apt-get install --no-install-recommends -y \
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

# Point the Vulkan loader at the NVIDIA ICD for Holoviz. Validation layers and
# loader debugging are opt-in at run time (e.g. -e VK_LOADER_DEBUG=all).
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Set PYTHONPATH to include ROS2 packages
ENV PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:/opt/nvidia/holoscan/python/lib

# Create necessary directories for Vulkan
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/vulkan/explicit_layer.d

# Create a proper NVIDIA Vulkan ICD configuration
RUN echo '{"file_format_version": "1.0.1", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4.329"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json

COPY <<'EOF' /usr/bin/entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
exec "$@"
EOF
RUN chmod +x /usr/bin/entrypoint.sh

ENTRYPOINT ["/usr/bin/entrypoint.sh"]

# Set up deps directory and clone holoscan-sensor-bridge
RUN mkdir -p /deps

# Clone the public Holoscan Sensor Bridge repository, pinned to a release
# tag for reproducible builds.
ARG HOLOLINK_REF=2.6.0
WORKDIR /deps
RUN git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git \
    && git -C holoscan-sensor-bridge checkout ${HOLOLINK_REF}

# Build the holoscan-sensor-bridge C++ libraries.
# HOLOLINK_BUILD_ROCE=OFF: this application uses the Linux socket receiver over
# the Jetson Thor MGBE ports, so no ConnectX NIC / RDMA (ibverbs) support is needed.
WORKDIR /deps/holoscan-sensor-bridge
RUN mkdir -p build
WORKDIR /deps/holoscan-sensor-bridge/build
RUN cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DHOLOLINK_BUILD_ROCE=OFF && make -j$(nproc)

# Install the holoscan-sensor-bridge Python package
WORKDIR /deps/holoscan-sensor-bridge/python
RUN pip3 install -e .

# Install the wrapper-pinned holoscan-cli; the holohub CLI runs `holoscan build`
# inside this container. Copying only the wrapper keeps this layer cached until
# the pin changes.
COPY --chmod=755 holohub /tmp/scripts/
RUN /tmp/scripts/holohub env-info

# Set default working directory
WORKDIR /workspace
