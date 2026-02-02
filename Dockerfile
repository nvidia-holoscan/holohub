# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG ENABLE_APT_CACHING=true

# Configure APT caching behavior
RUN if [ "${ENABLE_APT_CACHING}" = "true" ]; then \
        echo "APT Caching enabled..."; \
        DOCKER_CLEAN_CONF="/etc/apt/apt.conf.d/docker-clean"; \
        if [ -f "${DOCKER_CLEAN_CONF}" ]; then \
            mv "${DOCKER_CLEAN_CONF}" "${DOCKER_CLEAN_CONF}.disabled"; \
        fi; \
        echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
            > /etc/apt/apt.conf.d/99-keep-archives; \
    else \
        echo "APT Caching disabled..."; \
        rm -f /etc/apt/apt.conf.d/docker-clean; \
    fi

# --------------------------------------------------------------------------
#
# Set up prerequisites to run HoloHub CLI
#
# --------------------------------------------------------------------------
FROM base AS holohub-cli-prerequisites

# Install python3 if not present (needed for holohub CLI)
ARG PYTHON_VERSION=python3
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holohub-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holohub-apt-lib-$TARGETARCH-$GPU_TYPE \
    if ! command -v python3 >/dev/null 2>&1; then \
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
#
# Use HoloHub CLI to set up common packages for developing with Holoscan SDK
#
# --------------------------------------------------------------------------
FROM holohub-cli-prerequisites AS holohub-cli

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

# --------------------------------------------------------------------------
#
# Set up common packages for advanced development with
# Holoscan SDK Flow Benchmarking performance tools
#
# --------------------------------------------------------------------------
FROM holohub-cli AS benchmarking-setup

ARG CMAKE_BUILD_TYPE=Release

# For benchmarking
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holohub-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holohub-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt update \
    && apt install --no-install-recommends -y \
        libcairo2-dev \
        libgirepository1.0-dev \
        gobject-introspection \
        libgtk-3-dev \
        libcanberra-gtk-module \
        graphviz

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=holohub-pip-cache-$TARGETARCH-$GPU_TYPE \
    pip install meson

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=holohub-pip-cache-$TARGETARCH-$GPU_TYPE \
    if ! grep -q "VERSION_ID=\"22.04\"" /etc/os-release; then \
        pip install setuptools; \
    fi
COPY benchmarks/holoscan_flow_benchmarking/requirements.txt /tmp/benchmarking_requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked,id=holohub-pip-cache-$TARGETARCH-$GPU_TYPE \
    pip install -r /tmp/benchmarking_requirements.txt
ENV PYTHONPATH=/workspace/holohub/benchmarks/holoscan_flow_benchmarking

# --------------------------------------------------------------------------
#
# Set up common packages for developing with Yuan Qcap
#
# --------------------------------------------------------------------------
FROM holohub-cli AS yuan-qcap

# Qcap dependency
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holohub-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holohub-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt update \
    && apt install --no-install-recommends -y \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgles2 \
        libopengl0

# --------------------------------------------------------------------------
#
# Set up common packages for developing with RTI Connext DDS
#
# --------------------------------------------------------------------------
FROM holohub-cli AS dds

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holohub-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holohub-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt update \
    && apt install --no-install-recommends -y \
        openjdk-21-jre
RUN echo 'export JREHOME=$(readlink /etc/alternatives/java | sed -e "s/\/bin\/java//")' >> /etc/bash.bashrc

# --------------------------------------------------------------------------
#
# Set up packages for developing with AJA Capture Cards
#
# --------------------------------------------------------------------------
FROM holohub-cli AS holohub-aja

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=holohub-apt-cache-$TARGETARCH-$GPU_TYPE \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=holohub-apt-lib-$TARGETARCH-$GPU_TYPE \
    apt update \
    && apt install --no-install-recommends -y \
        libudev-dev

# --------------------------------------------------------------------------
#
# Default development stage. Use "--target <layer>" to build a different stage above
# as the environment for project development.
#
# --------------------------------------------------------------------------
FROM holohub-cli AS holohub-dev
