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
# Prerequisites: normalize the base image into one that can run
# the consolidated HoloHub CLI. See the root Dockerfile for the
# rationale; this stage is the same contract.
############################################################
FROM ${BASE_IMAGE} AS holohub-cli-prerequisites

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=python3

# Do NOT apt-install ${PYTHON_VERSION}-pip on Ubuntu 24.04+: that pip is
# debian-managed and missing the RECORD file, so a transitive
# `pip>25.1.0` upgrade from holoscan-cli aborts with `Cannot uninstall
# pip 24.0, RECORD file not found`. Install pip via get-pip.py instead.
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} curl ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! python3 -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_VERSION}; \
    fi

# Ensure the consolidated `holoscan` CLI exists. Probe the public `holoscan
# --help` surface for a source-project subcommand (`build`) rather than just
# `command -v holoscan` — the legacy packaging-only `holoscan` (holoscan-cli
# <= 4.2.0) in older Holoscan SDK images stays on PATH but doesn't register
# the source-project commands this stage relies on.
#
# Hard-pinned to the consolidated holoscan-cli build on TestPyPI for the
# consolidation rollout (matches the ./holohub wrapper pin). Plain pip only —
# `git+` specs and local-source installs are intentionally unsupported.
RUN if ! holoscan --help 2>/dev/null | grep -qw build; then \
        python3 -m pip install --upgrade \
            --index-url https://test.pypi.org/simple/ \
            "holoscan-cli==4.3.0a26390596878"; \
    fi

############################################################
# HoloHub CLI: stage the in-tree wrapper and smoke-test.
############################################################
FROM holohub-cli-prerequisites AS holohub-cli

ARG CMAKE_BUILD_TYPE=Release

# Stage the wrapper + utilities/ helpers `holohub setup` consumes (e.g.
# utilities/setup/*.sh, utilities/requirements.*.txt) so the raw-base
# fallback below has them when it runs `/tmp/scripts/holohub setup`.
RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub

# --------------------------------------------------------------------------
#
# Set up GStreamer-specific dependencies
#
# --------------------------------------------------------------------------
RUN holoscan version \
    && test -x /tmp/scripts/holohub

# Run HoloHub setup when this image is built from a raw base. The prepared
# base layer (utilities/docker/Dockerfile.holohub-base) already runs setup
# and drops the autocomplete marker; reuse that marker as the guard so the
# standard path stays a no-op while `--base-img` (which bypasses the
# prepared layer) still picks up recommended packages and autocomplete.
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

# Enable autocomplete if the prepared-base layer (or external setup) installed
# it — raw `--base-img` builds skip the prepared base entirely, so guard the
# source on file existence to avoid spamming "No such file or directory" into
# every interactive shell.
RUN echo "[ -f /etc/bash_completion.d/holohub_autocomplete ] && . /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# Set default Holohub data directory
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data
