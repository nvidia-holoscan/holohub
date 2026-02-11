# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG TARGETARCH
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# Set up prerequisites to run HoloHub CLI
#
# --------------------------------------------------------------------------
FROM base AS holohub-cli-prerequisites

# Install python3 if not present (needed for holohub CLI)
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
#
# Set up dependencies for developing with PVA-SDK and pva-solutions
#
# --------------------------------------------------------------------------

# Install deb packages for the PVA-SDK and pva-solutions.
# The pva-sdk packages are expected to be the local installer meta packages named
# pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-local-*_${PVA_SDK_VERSION}_all.deb and should be placed in
# the operators/pva_radar/deps/ directory.
# Each local installer contains several debian packages. Layers defined downstream will install a
# subset of these packages depending on the target architecture.
# The pva-solutions deb packages should be built offline following the instructions from the pva-solutions
# documentation. Include both L4T (arm64) and NATIVE (amd64) packages in the same location.
RUN mkdir -p /tmp/pva_deps
COPY operators/pva_radar/deps/* /tmp/pva_deps/
RUN for deb in /tmp/pva_deps/*.deb; do dpkg -i $deb; done
# Unpack the pva-solutions source code tarball into the container workspace.
ARG PVA_SOLUTIONS_VERSION=0.4.0
RUN mkdir -p /workspace/pva-solutions && \
    tar -xzf /tmp/pva_deps/pva-solutions-${PVA_SOLUTIONS_VERSION}.tar.gz --strip-components=1 -C /workspace/pva-solutions
RUN rm -rf /tmp/pva_deps
# Sync CVCUDA sources for the NVCV types headers.
# NOTE: Normally cmake handles this step when building pva-solutions, but since we only use a few source files and headers
# from pva-solutions for pva_radar, we do it manually to avoid running the full cmake configure step.
ARG CVCUDA_VERSION_TAG=v0.14.0-beta
RUN cd /workspace/pva-solutions/3rdparty && \
    git clone https://github.com/CVCUDA/CV-CUDA.git --branch ${CVCUDA_VERSION_TAG} cvcuda


# --------------------------------------------------------------------------
#
# Set up PVA prerequisites depending on the target architecture
#
# --------------------------------------------------------------------------

# Set up PVA-SDK development environment on amd64 host PC (x86_64)
# This enables the PVA emulator environment for prototyping.
FROM holohub-cli-prerequisites AS holohub-pva-prerequisites-amd64
ARG PVA_SDK_MAJOR_MINOR_VERSION=2.8
RUN apt-get update \
    && apt-get install -y \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-dev \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-gen2-dev \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-gen3-dev \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-debug-tools \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-native-pvaumd \
    && rm -rf /var/lib/apt/lists/*


# Set up the PVA host runtime on Jetson target device (arm64)
FROM holohub-cli-prerequisites AS holohub-pva-prerequisites-arm64
ARG PVA_SDK_MAJOR_MINOR_VERSION=2.8
ARG PVA_SDK_VERSION=2.8.0
RUN apt-get update \
    && apt-get install -y \
    pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-l4t \
    && rm -rf /var/lib/apt/lists/*

# Extract PVA-SDK header files from the amd64 dev package manually
# NOTE: We need this step because building on arm64 target is not officially supported by PVA-SDK and there is no arm64
# package that includes cupva headers. We use a few cupva runtime APIs in our operators so we need these headers.
RUN mkdir -p /tmp/pva-sdk-dev-files \
    && dpkg-deb -x /var/pva-sdk-local-core/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}-dev_${PVA_SDK_VERSION}_amd64.deb /tmp/pva-sdk-dev-files/ \
    && cp -r /tmp/pva-sdk-dev-files/opt/nvidia/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}/include /opt/nvidia/pva-sdk-${PVA_SDK_MAJOR_MINOR_VERSION}/ \
    && rm -rf /tmp/pva-sdk-dev-files


# --------------------------------------------------------------------------
#
# Use HoloHub CLI to set up common packages for developing with Holoscan SDK
#
# --------------------------------------------------------------------------
FROM holohub-pva-prerequisites-${TARGETARCH} AS holohub-cli

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc


# --------------------------------------------------------------------------
#
# Set up common packages for advanced development with
# Holoscan SDK Flow Benchmarking performance tools
#
# --------------------------------------------------------------------------
FROM holohub-cli AS benchmarking-setup

ARG CMAKE_BUILD_TYPE=Release

# For benchmarking
RUN apt update \
    && apt install --no-install-recommends -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    libgtk-3-dev \
    libcanberra-gtk-module \
    graphviz

RUN pip install meson

RUN if ! grep -q "VERSION_ID=\"22.04\"" /etc/os-release; then \
    pip install setuptools; \
    fi
COPY benchmarks/holoscan_flow_benchmarking/requirements.txt /tmp/benchmarking_requirements.txt
RUN pip install -r /tmp/benchmarking_requirements.txt
ENV PYTHONPATH=/workspace/holohub/benchmarks/holoscan_flow_benchmarking

# --------------------------------------------------------------------------
#
# Set up common packages for developing with Yuan Qcap
#
# --------------------------------------------------------------------------
FROM holohub-cli AS yuan-qcap

# Qcap dependency
RUN apt update \
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

RUN apt update \
    && apt install --no-install-recommends -y \
    openjdk-21-jre
RUN echo 'export JREHOME=$(readlink /etc/alternatives/java | sed -e "s/\/bin\/java//")' >> /etc/bash.bashrc

# --------------------------------------------------------------------------
#
# Set up packages for developing with AJA Capture Cards
#
# --------------------------------------------------------------------------
FROM holohub-cli AS holohub-aja

RUN apt update \
    && apt install --no-install-recommends -y \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
#
# Default development stage. Use "--target <layer>" to build a different stage above
# as the environment for project development.
#
# --------------------------------------------------------------------------
FROM holohub-cli AS holohub-dev
