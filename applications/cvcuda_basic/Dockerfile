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
ARG CVCUDA_TAG=v0.4.0-beta
ARG CVCUDA_RELEASE_BRANCH=release_v0.4.x
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/nvidia

# install gcc-11, g++-11
RUN apt update \
    && apt install --no-install-recommends -y \
      software-properties-common="0.99.*"

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt install -y gcc-${GCC_VERSION} g++-${GCC_VERSION}

# download patch for building CV-CUDA with CUDA 11.6
RUN curl -S -L -o cvcuda_cuda11_6.patch  "https://gist.githubusercontent.com/grlee77/ba5d1449ae37ff2e13c7ee93710fe482/raw/f335c08d00fb6e6efeb94e99889fd15957f01d47/cvcuda_cuda11_6.patch"

# download and patch CV-CUDA
RUN git clone https://github.com/CVCUDA/CV-CUDA \
    && mv cvcuda_cuda11_6.patch CV-CUDA \
    && cd CV-CUDA \
    && git checkout ${CVCUDA_TAG} \
    && git submodule update --init \
    && git apply cvcuda_cuda11_6.patch

# replace broken stub binaries in the v0.3.1-beta tag
# (https://github.com/CVCUDA/CV-CUDA/issues/26)
RUN cd CV-CUDA/src/util/stubs \
    && rm *.so \
	&& curl -S -L -o libdl-2.17_stub.so https://github.com/CVCUDA/CV-CUDA/raw/${CVCUDA_RELEASE_BRANCH}/src/util/stubs/libdl-2.17_stub.so \
	&& curl -S -L -o libpthread-2.17_stub.so https://github.com/CVCUDA/CV-CUDA/raw/${CVCUDA_RELEASE_BRANCH}/src/util/stubs/libpthread-2.17_stub.so \
	&& curl -S -L -o librt-2.17_stub.so https://github.com/CVCUDA/CV-CUDA/raw/${CVCUDA_RELEASE_BRANCH}/src/util/stubs/librt-2.17_stub.so

############################################################
# CV-CUDA Builder
############################################################

FROM cvcuda-downloader as cvcuda-builder
WORKDIR /opt/nvidia/CV-CUDA

# compile CV-CUDA
RUN bash ./ci/build.sh

# create and install the Debian packages
RUN cd build-rel \
	&& cpack -G DEB . \
    && dpkg -i nvcv*.deb 

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
