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

# Install the wrapper-pinned holoscan-cli; copying only the wrapper keeps
# this layer cached until the pin changes.
COPY --chmod=755 holoscan_camera /tmp/scripts/
RUN /tmp/scripts/holoscan_camera env-info

# TODO: add module-specific runtime / build dependencies below.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgtest-dev \
        clang-format \
        ninja-build \
        xvfb \
        ccache \
    && rm -rf /var/lib/apt/lists/*
