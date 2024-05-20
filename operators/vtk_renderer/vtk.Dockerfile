# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE=holohub:ngc-v1.0.3-dgpu
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt install -y \
        libglvnd-dev \
        ninja-build

# Create directories
RUN mkdir -p /tmp/vtk/ && \
    mkdir -p /opt/vtk/

WORKDIR /tmp/vtk
RUN curl --remote-name https://gitlab.kitware.com/vtk/vtk/-/archive/v9.3.0/vtk-v9.3.0.tar.gz && \
    tar -xvzf vtk-v9.3.0.tar.gz && \
    rm vtk-v9.3.0.tar.gz && \
    cmake -GNinja -S vtk-v9.3.0 -B vtk-build \
        -DVTK_MODULE_ENABLE_RenderingCore=YES \
        -DVTK_MODULE_ENABLE_RenderingFFMPEGOpenGL2=YES \
        -DVTK_MODULE_ENABLE_RenderingOpenGL2=YES && \
    cmake --build vtk-build && \
    cmake --install vtk-build --prefix=/opt/vtk && \
    rm -rf vtk-build
