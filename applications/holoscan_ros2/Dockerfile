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

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-dgpu AS base

# Install basic dependencies
RUN apt update && apt install -y \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Jazzy repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy
RUN apt update && apt install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-base \
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


# Set PYTHONPATH to include ROS2 packages - this will be appended to by isaac-hub script
ENV PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:/opt/nvidia/holoscan/python/lib

RUN echo -e '#!/bin/bash \n \
set -e \n \
source /opt/ros/jazzy/setup.bash \n \
exec "$@"' > /usr/bin/entrypoint.sh && chmod +x /usr/bin/entrypoint.sh

ENTRYPOINT ["/usr/bin/entrypoint.sh"]
