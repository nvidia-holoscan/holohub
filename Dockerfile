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

############################################################
# Prerequisites: normalize the base image (SDK / CUDA / plain Ubuntu) into
# one that can run the HoloHub CLI. Each step is a no-op when the base
# already provides the requirement.
############################################################
FROM ${BASE_IMAGE} AS holohub-cli-prerequisites

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=python3

# 1. Ensure python3 + pip exist. Install pip via get-pip.py, NOT apt
#    `${PYTHON_VERSION}-pip`: the debian-managed pip lacks a RECORD file, so a
#    transitive `pip>25.1.0` upgrade from holoscan-cli aborts with
#    "Cannot uninstall pip 24.0, RECORD file not found".
#    Guard on ${PYTHON_VERSION} (not bare python3) so a PYTHON_VERSION override
#    on a base that already ships python3 still installs the requested
#    interpreter via deadsnakes when the base apt repositories do not carry it.
RUN if ! command -v "${PYTHON_VERSION}" >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            software-properties-common curl ca-certificates gpg-agent \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} \
        && apt purge -y \
            python3-pip \
            software-properties-common \
        && apt-get autoremove --purge -y \
        && rm -rf /var/lib/apt/lists/*; \
    fi
RUN update-alternatives --install /usr/bin/python python "/usr/bin/${PYTHON_VERSION}" 100 \
    && if [ "${PYTHON_VERSION}" != "python3" ]; then \
        update-alternatives --install /usr/bin/python3 python3 "/usr/bin/${PYTHON_VERSION}" 100; \
    fi
RUN if ! command -v curl >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y curl ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! "${PYTHON_VERSION}" -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | "${PYTHON_VERSION}"; \
    fi

# 2. Ensure the `holoscan` CLI exists. Probe `holoscan --help` for a
#    source-project command (`build`) so an incompatible installed CLI does not
#    satisfy the check. HOLOSCAN_CLI_INSTALL_ARGS (the wrapper forwards it as a
#    build-arg) selects the package and any pip options.
ARG HOLOSCAN_CLI_INSTALL_ARGS=--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0
RUN if ! holoscan --help 2>/dev/null | grep -qw build; then \
        "${PYTHON_VERSION}" -m pip install \
            ${HOLOSCAN_CLI_INSTALL_ARGS}; \
    fi

############################################################
# HoloHub CLI: stage the in-tree wrapper and smoke-test.
############################################################
FROM holohub-cli-prerequisites AS holohub-cli

ARG CMAKE_BUILD_TYPE=Release

# 3. Stage the in-tree `holohub` wrapper + the utilities/ helpers `holohub
#    setup` needs, so the wrapper is callable inside the container.
RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub

# 4. Smoke-test: both the CLI and the wrapper are present and usable.
RUN holoscan version \
    && test -x /tmp/scripts/holohub

# 5. Run `holohub setup` only on raw-base builds. The prepared base already
#    ran it and left the autocomplete marker, so guard on that marker to keep
#    the standard (prepared-base) path a no-op.
RUN [ -f /etc/bash_completion.d/holohub_autocomplete ] \
    || (/tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*)
RUN grep -qxF "[ -f /etc/bash_completion.d/holohub_autocomplete ] && . /etc/bash_completion.d/holohub_autocomplete" /etc/bash.bashrc \
    || echo "[ -f /etc/bash_completion.d/holohub_autocomplete ] && . /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# 6. Mirror the prepared-base default data path so `--base-img` builds see the
#    same HOLOSCAN_INPUT_PATH.
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
