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

# Default-empty source stage for the consolidated holoscan-cli, mirroring
# utilities/docker/Dockerfile.holohub-base. Override with a real checkout via
# `--build-context holoscan-cli-src=<path>`; the host wrapper passes that
# automatically when HOLOSCAN_CLI_SOURCE is set so iterating on holoscan-cli
# against `--base-img` builds picks up the local checkout instead of falling
# back to PyPI.
FROM scratch AS holoscan-cli-src

############################################################
# Prerequisites: normalize the base image into one that can run
# the consolidated HoloHub CLI.
#
# BASE_IMAGE can be an SDK image (ships python + holoscan), a CUDA
# base image (ships nothing Python-related), or a plain Ubuntu image.
# Each conditional below is a no-op when the base already provides
# the requirement, so SDK builds stay cheap.
############################################################
FROM ${BASE_IMAGE} AS holohub-cli-prerequisites

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=python3
ARG HOLOSCAN_CLI_INSTALL_SPEC=holoscan-cli
# Extra pip flags injected via HOLOSCAN_CLI_DEFAULT_DOCKER_BUILD_ARGS by the
# host wrapper (e.g. `--pre --index-url https://test.pypi.org/simple/` while
# the consolidated CLI release is on TestPyPI). Empty by default — same
# contract as utilities/docker/Dockerfile.holohub-base.
ARG HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS=

# 1. Ensure python3 + pip exist.
#
# Do NOT apt-install ${PYTHON_VERSION}-pip on Ubuntu 24.04+: that pip is
# debian-managed and missing the RECORD file, so when ``holoscan-cli``
# transitively requests ``pip>25.1.0`` the upgrade aborts with
# ``Cannot uninstall pip 24.0, RECORD file not found``. Install pip via
# get-pip.py instead — it lands under /usr/local and is freely upgradeable.
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

# 2. Ensure `git` is available before bootstrapping holoscan-cli. Pip needs
#    it for `git+` install specs and the local-source build path also needs
#    it (poetry-dynamic-versioning -> dunamai reads the version from VCS
#    metadata, fails with "Unable to find 'git' program" otherwise).
#    SDK base images already ship `git`, so this stays a no-op for the
#    standard prepared-base path; only raw bases (ubuntu:22.04, cuda:*-base)
#    actually trigger the install.
RUN if ! holoscan --help 2>/dev/null | grep -qw build \
        && ! command -v git >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y git \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# 3. Ensure the consolidated `holoscan` CLI exists. Skip the pip install
#    only when the base already ships a *post-consolidation* CLI — the
#    legacy packaging-only `holoscan` (holoscan-cli <= 4.2.0) that older
#    Holoscan SDK images bundle stays on PATH and still answers `holoscan
#    version`, so neither `command -v holoscan` nor a version probe can
#    tell it apart. Grep the public `holoscan --help` surface for a
#    source-project subcommand (`build`) that only the consolidated CLI
#    registers, matching the check used by
#    utilities/docker/Dockerfile.holohub-base.
#
#    When the `holoscan-cli-src` build-context is non-empty (the wrapper
#    sets it from `HOLOSCAN_CLI_SOURCE`), install from that checkout in
#    preference to `HOLOSCAN_CLI_INSTALL_SPEC` so in-container and host
#    CLIs stay in lockstep.
RUN --mount=type=bind,from=holoscan-cli-src,target=/tmp/holoscan-cli-src \
    if ! holoscan --help 2>/dev/null | grep -qw build; then \
        if [ -f /tmp/holoscan-cli-src/pyproject.toml ]; then \
            echo "Installing consolidated holoscan-cli from local source build-context"; \
            cp -a /tmp/holoscan-cli-src /tmp/holoscan-cli-src-writable; \
            python3 -m pip install --no-cache-dir /tmp/holoscan-cli-src-writable; \
            rm -rf /tmp/holoscan-cli-src-writable; \
        else \
            python3 -m pip install --upgrade ${HOLOSCAN_CLI_INSTALL_EXTRA_FLAGS} "${HOLOSCAN_CLI_INSTALL_SPEC}"; \
        fi; \
    fi

############################################################
# HoloHub CLI: stage the in-tree wrapper and smoke-test.
############################################################
FROM holohub-cli-prerequisites AS holohub-cli

ARG CMAKE_BUILD_TYPE=Release

# 4. Stage /tmp/scripts/holohub plus the utilities/ helpers `holohub setup`
#    consumes (e.g. utilities/setup/*.sh, utilities/requirements.*.txt) so
#    the in-tree wrapper is callable inside the container regardless of base.
RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub

# 5. Smoke-test: both the CLI and the wrapper are present and usable.
RUN holoscan version \
    && test -x /tmp/scripts/holohub

# 6. Run HoloHub setup when this image is built from a raw base. The prepared
#    base layer (utilities/docker/Dockerfile.holohub-base) already runs setup
#    and drops the autocomplete marker; reuse that marker as the guard so the
#    standard path stays a no-op while `--base-img` (which bypasses the
#    prepared layer) still picks up recommended packages and autocomplete.
RUN [ -f /etc/bash_completion.d/holohub_autocomplete ] \
    || (/tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*)

# 7. Mirror the default data path from the prepared-base layer so apps that
#    read `HOLOSCAN_INPUT_PATH` see the same value whether or not callers
#    skipped the prepared base via `--base-img`.
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
