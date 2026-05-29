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
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} curl ca-certificates \
        && rm -rf /var/lib/apt/lists/* \
        && if [ "${PYTHON_VERSION}" != "python3" ]; then \
            update-alternatives --install /usr/bin/python3 python3 "/usr/bin/${PYTHON_VERSION}" 100; \
        fi; \
    fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! python3 -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_VERSION}; \
    fi

# 2. Ensure the `holoscan` CLI exists. The legacy packaging-only CLI
#    (holoscan-cli <= 4.2.0) is also on PATH, so probe `holoscan --help` for a
#    source-project command (`build`) only the new CLI has. Two-step install
#    (matches CI): the wheel from TestPyPI (--no-deps), then deps from PyPI so a
#    TestPyPI mirror can't shadow a PyPI dep. HOLOSCAN_CLI_INSTALL_SPEC (the
#    wrapper forwards it as a build-arg) selects the version; default is the
#    pinned build. Drop to a bare `pip install holoscan-cli` once it's on PyPI.
ARG HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli==4.3.0a26390596878
RUN if ! holoscan --help 2>/dev/null | grep -qw build; then \
        python3 -m pip install --pre --no-deps \
            --index-url https://test.pypi.org/simple/ "${HOLOSCAN_CLI_INSTALL_SPEC}" \
        && python3 -m pip install \
            --index-url https://pypi.org/simple/ "${HOLOSCAN_CLI_INSTALL_SPEC}"; \
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
