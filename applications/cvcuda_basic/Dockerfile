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

############################################################
# CV-CUDA Downloader
############################################################
FROM ${BASE_IMAGE} as cvcuda-downloader

ARG GCC_VERSION=11
ARG CVCUDA_TAG=v0.7.0-beta
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/nvidia

# install gcc-11, g++-11
# proper checkout of CV-CUDA stubs also requires git-lfs
RUN apt update \
    && apt install --no-install-recommends -y \
      software-properties-common="0.99.*" \
      git-lfs

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt install -y gcc-${GCC_VERSION} g++-${GCC_VERSION}

# download CV-CUDA
RUN git clone --branch ${CVCUDA_TAG} --depth 1 https://github.com/CVCUDA/CV-CUDA \
    && cd CV-CUDA \
    && git submodule update --init

############################################################
# CV-CUDA Builder
############################################################

FROM cvcuda-downloader as cvcuda-builder
WORKDIR /opt/nvidia/CV-CUDA

# compile CV-CUDA
RUN bash ./ci/build.sh

# create and install the Debian packages
# (skip cvcuda-tests as it requires an additional python3-pytest dependency)
RUN cd build-rel \
    && cpack -G DEB . \
    && rm cvcuda-tests*.deb \
    && dpkg -i cvcuda*.deb \
    && find . -name "*.whl" -exec pip install {} \;

############################################################
# Base (final)
############################################################
FROM cvcuda-builder as base

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
