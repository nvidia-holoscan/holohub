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
FROM ${BASE_IMAGE} AS cvcuda-downloader

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

FROM cvcuda-downloader AS cvcuda-builder
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
FROM cvcuda-builder AS base
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update \
    && apt install --no-install-recommends -y \
        lsb-release

# Add librealsense apt source.
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" \
    | tee /etc/apt/sources.list.d/librealsense.list

# Install librealsense.
RUN apt update \
    && apt install --no-install-recommends -y \
        librealsense2-dev \
        libudev-dev \
        libusb-1.0-0-dev \
        libnvidia-gl-535 mesa-utils libgl1-mesa-dev

## this should probably be another build-layer ...

# install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
#ENV PATH="${PATH}:/home/holoscan/.cargo/bin"
ENV PATH="${PATH}:${HOME}/.cargo/bin"
RUN rustup update

RUN pip install conan && \
    conan profile detect && \
    conan remote add camposs "https://conan.campar.in.tum.de/artifactory/api/conan/conan-camposs"

WORKDIR /opt/tcn/dependencies
ADD applications/narvis_simple_capture/conan/holoscan_dependencies /opt/tcn/dependencies/holoscan_dependencies
ADD applications/narvis_simple_capture/conan/conanfile.py /opt/tcn/dependencies/conanfile.py
ADD applications/narvis_simple_capture/conan/relocatable_app.py /opt/tcn/dependencies/relocatable_app.py

WORKDIR /opt/tcn/dependencies/holoscan_dependencies
RUN conan export --user camposs --channel stable .

WORKDIR /opt/tcn/dependencies
RUN conan install . --build missing --build outdated --deployer=/opt/tcn/dependencies/relocatable_app.py --deployer-folder=/opt/tcn/3rdparty -c tools.system.package_manager:mode=install
ENV PATH="/opt/tcn/3rdparty/direct_deploy/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/tcn/3rdparty/direct_deploy/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="/opt/tcn/3rdparty/direct_deploy"

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

RUN apt update \
    && apt install --no-install-recommends -y \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

