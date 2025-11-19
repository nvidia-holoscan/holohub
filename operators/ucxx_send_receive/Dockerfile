# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# Use HoloHub CLI to set up common packages for developing with Holoscan SDK
#
# --------------------------------------------------------------------------
FROM base AS holohub-cli

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# Set default Holohub data directory
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

############################################################
# Flatbuffers-builder
############################################################
FROM holohub-cli AS flatbuffers-builder
ARG FLATBUFFERS_VERSION=25.2.10

WORKDIR /opt/flatbuffers

# Clone
RUN git clone --branch v${FLATBUFFERS_VERSION} https://github.com/google/flatbuffers.git src

# Patch
RUN --mount=type=bind,source=operators/ucxx_send_receive/patches/flatbuffers,target=/opt/flatbuffers/patches \
    cd src && git apply ../patches/support_native_type_table.patch

# Build
RUN mkdir -p build
RUN cmake -S src -B build -G Ninja \
      -D FLATBUFFERS_BUILD_TESTS=OFF \
      -D FLATBUFFERS_BUILD_FLATHASH=OFF \
      -D FLATBUFFERS_BUILD_FLATC=ON \
      -D FLATBUFFERS_BUILD_FLATLIB=ON \
      -D FLATBUFFERS_STATIC_FLATC=ON \
      -D FLATBUFFERS_STRICT_MODE=ON
RUN cmake --build build -j
RUN cmake --install build --prefix /opt/flatbuffers/${FLATBUFFERS_VERSION}

ENV CMAKE_PREFIX_PATH=/opt/flatbuffers/${FLATBUFFERS_VERSION}:${CMAKE_PREFIX_PATH}
