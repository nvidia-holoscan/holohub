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

FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive


# --------------------------------------------------------------------------
#
# Holohub run setup
#

RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release


# --------------------------------------------------------------------------
#
# VTK 9.3.1  Build
#

ARG VTK_VERSION=9.3.1

WORKDIR /opt/vtk

RUN apt update &&  apt install  --no-install-recommends -y libglvnd-dev
RUN  wget -q https://www.vtk.org/files/release/9.3/VTK-${VTK_VERSION}.tar.gz -P src && cd src && tar xzvf VTK-${VTK_VERSION}.tar.gz && \
     mkdir -p /opt/vtk/build && \
     cd /opt/vtk/build  && \
     cmake  -DBUILD_TESTING:BOOL=OFF \
            -DCMAKE_INSTALL_PREFIX=/opt/vtk/${VTK_VERSION} \
            /opt/vtk/src/VTK-${VTK_VERSION} && \
     make -j 8 && \
     make install && \
     make clean

RUN rm -rf /opt/vtk/build &&  rm -rf /opt/vtk/src

# --------------------------------------------------------------------------
#
# VTK environment setup
#
ENV VTK_INSTALL=/opt/vtk/${VTK_VERSION}
ENV PATH="${PATH}:${VTK_INSTALL}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VTK_INSTALL}"


# --------------------------------------------------------------------------
#
# GLFW  & GLEW
#

RUN apt install -y libglfw3-dev libglew-dev 


WORKDIR /workspace/holohub