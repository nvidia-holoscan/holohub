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

# Default-empty source stage for the consolidated holoscan-cli. Override with a
# real checkout via:
#   docker buildx build --build-context holoscan-cli-src=/path/to/holoscan-cli ...
# The holoscan-cli host CLI passes this automatically when HOLOSCAN_CLI_SOURCE
# is set in the environment.
FROM scratch AS holoscan-cli-src

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

# Install the consolidated Holoscan Platform CLI. Three install sources are
# supported, in order of precedence:
#   1. Local checkout exposed via `--build-context holoscan-cli-src=<path>`.
#      The host CLI passes this automatically when HOLOSCAN_CLI_SOURCE is set.
#   2. The HOLOSCAN_CLI_INSTALL_SPEC build arg, optionally combined with
#      HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS for index URLs / pre-release flags.
#      Examples:
#        --build-arg HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli==4.3.0
#        --build-arg HOLOSCAN_CLI_INSTALL_SPEC=git+https://github.com/nvidia-holoscan/holoscan-cli.git@main
#        --build-arg HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli==4.3.0a26027723520 \
#        --build-arg "HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS=--pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/"
#   3. Default value `holoscan-cli` (the published PyPI release).
# Provides the `holoscan` console script used by the in-container recursion
# (`holoscan build/run/install <project> --local`). The local-source path
# copies to a writable temp dir before installing because BuildKit bind mounts
# are read-only and pip's PEP 517 build writes transient artifacts into the
# source tree (e.g. poetry-dynamic-versioning version metadata).
ARG HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli
ARG HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS=
RUN --mount=type=bind,from=holoscan-cli-src,target=/tmp/holoscan-cli-src \
    if [ -f /tmp/holoscan-cli-src/pyproject.toml ]; then \
        echo "Installing consolidated holoscan-cli from local source build-context"; \
        cp -a /tmp/holoscan-cli-src /tmp/holoscan-cli-src-writable; \
        python3 -m pip install --no-cache-dir /tmp/holoscan-cli-src-writable; \
        rm -rf /tmp/holoscan-cli-src-writable; \
    else \
        echo "Installing holoscan-cli from spec: ${HOLOSCAN_CLI_INSTALL_SPEC} (extra flags: ${HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS:-none})"; \
        python3 -m pip install --no-cache-dir ${HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS} "${HOLOSCAN_CLI_INSTALL_SPEC}"; \
    fi

# Fail the build fast if the installed `holoscan-cli` is pre-consolidation
# (e.g. PyPI 4.0.0). That release has the `holoscan` console script and a
# `version` subcommand, so a smoke that only runs `holoscan version` passes
# silently; the discriminator is the source-project dispatch table that the
# in-container recursion (`holoscan build/run/install <project> --local`)
# relies on.
RUN holoscan version \
    && python3 -c "from holoscan_cli.__main__ import PROJECT_COMMANDS; assert {'build','run','list','install'} <= set(PROJECT_COMMANDS), f'consolidated CLI missing commands: {set(PROJECT_COMMANDS)}'"

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
