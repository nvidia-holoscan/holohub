# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for Holoscan Camera.
#
# Extends the Holoscan SDK image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/holoscan_camera when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holoscan_camera run-container
#
# Manual build (without the CLI):
#   docker build --build-arg BASE_IMAGE=holoscan-sdk-build-$(uname -m):latest \
#       -t holoscan-holoscan-camera .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# JetPack / L4T version that the SIPL API headers are sourced from.
# Default is the minimum version required by this project (SIPL API v2, JetPack 7.x / L4T R39+).
# Override with --build-arg L4T_VERSION=<ver> if a specific release is needed.
ARG L4T_VERSION=39.2.0
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_PATCH_VER="${L4T_VERSION#*.}"
# L4T minor version only (e.g. "2.0" → "2") — used for L4T APT repo label.
ENV L4T_MIN_VER="${L4T_MIN_PATCH_VER%%.*}"

# Install the wrapper-pinned holoscan-cli; copying only the wrapper keeps
# this layer cached until the pin changes.
COPY --chmod=755 holoscan_camera /tmp/scripts/
RUN /tmp/scripts/holoscan_camera env-info

# Module-specific build dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake \
        curl \
        libgtest-dev \
        clang-format \
        ninja-build \
        xvfb \
        ccache \
    && rm -rf /var/lib/apt/lists/*

# SIPL API headers (JetPack R${L4T_VERSION}).
# Downloaded from developer.nvidia.com — single source of truth for the
# container build. SIPL runtime libs are NOT baked in; mount them from the
# host BSP at container run time. --allow-shlib-undefined in CMakeLists.txt
# lets the linker proceed when they are not present at build time.
RUN curl -fsSL -O \
        "https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_PATCH_VER}/release/Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2" \
    && tar xjf "Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2" -C / \
    && rm "Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2" \
    && if [ -f /usr/src/jetson_sipl_api/sipl/CMakeLists.txt ]; then \
           cd /usr/src/jetson_sipl_api/sipl \
           && mkdir build && cd build \
           && cmake .. \
           && make -j"$(nproc)" \
           && make install \
           || echo "SIPL API cmake build skipped (BSP libs not present at build time)"; \
       fi \
    && echo "SIPL API headers installed to /usr/src/jetson_sipl_api"
