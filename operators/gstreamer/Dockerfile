# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}

############################################################
# Prerequisites: normalize the base image into one that can run the HoloHub
# CLI. See the root Dockerfile for the rationale; this stage is the same
# contract.
############################################################
FROM ${BASE_IMAGE} AS holohub-cli-prerequisites

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=python3

# Do NOT apt-install ${PYTHON_VERSION}-pip on Ubuntu 24.04+: that pip is
# debian-managed and missing the RECORD file, so a transitive
# `pip>25.1.0` upgrade from holoscan-cli aborts with `Cannot uninstall
# pip 24.0, RECORD file not found`. Install pip via get-pip.py instead.
# Guard on ${PYTHON_VERSION} (not bare python3) so a PYTHON_VERSION override on
# a base that already ships python3 still installs the requested interpreter;
# also (re)install when curl is missing, since the get-pip.py bootstrap below
# needs curl + CA roots even when python is already present.
RUN if ! command -v "${PYTHON_VERSION}" >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then \
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

# Ensure the `holoscan` CLI exists. The legacy packaging-only CLI (holoscan-cli
# <= 4.2.0) is also on PATH, so probe `holoscan --help` for a source-project
# command (`build`) only the new CLI has. Install the pinned release from NVIDIA
# PyPI until it is promoted to public PyPI. HOLOSCAN_CLI_INSTALL_SPEC (the
# wrapper forwards it) selects the version; default is the pinned build.
ARG HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli==4.3.0rc2
RUN if ! holoscan --help 2>/dev/null | grep -qw build; then \
        python3 -m pip install --pre \
            --index-url https://pypi.nvidia.com "${HOLOSCAN_CLI_INSTALL_SPEC}"; \
    fi

############################################################
# HoloHub CLI: stage the in-tree wrapper and smoke-test.
############################################################
FROM holohub-cli-prerequisites AS holohub-cli

ARG CMAKE_BUILD_TYPE=Release

# Stage the `holohub` wrapper + utilities/ helpers `holohub setup` needs, so
# the raw-base `holohub setup` fallback below can run.
RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub

# Smoke-test: CLI + wrapper present and usable.
RUN holoscan version \
    && test -x /tmp/scripts/holohub

# Run `holohub setup` only on raw-base builds. The prepared base already ran
# it and left the autocomplete marker, so guard on that marker.
RUN [ -f /etc/bash_completion.d/holohub_autocomplete ] \
    || (/tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*)

# Install GStreamer development packages and pkg-config
COPY operators/gstreamer/install_deps.sh /tmp/install_deps.sh
RUN /tmp/install_deps.sh

# Build tooling for C++ development and Python binding compilation
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        clang-format \
        pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tooling: pytest for tests, build + scikit-build-core for wheel packaging
RUN python3 -m pip install --no-cache-dir \
        pytest \
        pytest-timeout \
        build \
        scikit-build-core

# Source autocomplete only if present (raw `--base-img` builds skip the
# prepared base), to avoid "No such file or directory" in every shell.
RUN echo "[ -f /etc/bash_completion.d/holohub_autocomplete ] && . /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# Set default Holohub data directory
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
