# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# =============================================================================
# AI Surgical Video Application Container
#
# Stages:
#   holohub-cli  — base HoloHub CLI environment (replayer / aja modes)
#   holohub-hsb  — holohub-cli + HSB build & runtime dependencies (hsb mode)
#
# Build with the holohub CLI:
#   ./holohub run-container ai_surgical_video [--mode replayer|aja|hsb]
# =============================================================================

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# Set up prerequisites to run HoloHub CLI
# --------------------------------------------------------------------------
FROM base AS holohub-cli-prerequisites

ARG PYTHON_VERSION=python3
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            software-properties-common curl gpg-agent \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} \
        && apt purge -y \
            python3-pip \
            software-properties-common \
        && apt-get autoremove --purge -y \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON_VERSION} 100 \
        && if [ "${PYTHON_VERSION}" != "python3" ]; then \
            update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_VERSION} 100 \
            ; fi \
    ; fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! python3 -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_VERSION} \
    ; fi

# --------------------------------------------------------------------------
# Use HoloHub CLI to set up common packages for developing with Holoscan SDK
# --------------------------------------------------------------------------
FROM holohub-cli-prerequisites AS holohub-cli

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

# --------------------------------------------------------------------------
# Holoscan Sensor Bridge stage: system dependencies for building and running Holoscan Sensor
# Bridge.  The hololink Python module itself is built by the CMake subproject
# (FetchContent pulls the source; CMake builds the module).
#
# Build deps (needed by hololink's CMake / pybind11 extension):
#   cmake, ninja-build        — build system
#   python3-dev               — Python headers for the C extension
#   python3-setuptools        — pip wheel build support
#   libibverbs-dev            — InfiniBand verbs headers (roce_receiver et al.)
#   libnuma-dev               — NUMA awareness headers
#   libelf-dev                — ELF parsing (transitive dep of DOCA)
#   libfmt-dev                — fmt library headers
#   libyaml-cpp-dev           — YAML parsing headers
#   libdoca-sdk-gpunetio-dev  — DOCA GPUNetIO pkg-config module (gpu_roce_transceiver)
#   libdoca-sdk-verbs-dev     — DOCA verbs pkg-config module   (gpu_roce_transceiver)
#   libcurl4-openssl-dev      — libcurl for hsb_flasher tool
#
# Runtime deps (needed when the container runs the application):
#   libibverbs1, ibverbs-utils, rdma-core — RoCE / InfiniBand userspace stack
#   libnuma1                              — NUMA runtime
# --------------------------------------------------------------------------
FROM holohub-cli AS holohub-hsb

RUN apt-get update && apt-get install --no-install-recommends -y \
        cmake \
        ninja-build \
        python3-dev \
        python3-setuptools \
        libibverbs-dev \
        libibverbs1 \
        ibverbs-utils \
        libnuma-dev \
        libnuma1 \
        libelf-dev \
        libfmt-dev \
        libyaml-cpp-dev \
        libdoca-sdk-gpunetio-dev \
        libdoca-sdk-verbs-dev \
        libcurl4-openssl-dev \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display
ENV NVIDIA_VISIBLE_DEVICES=all
WORKDIR /workspace/holohub
