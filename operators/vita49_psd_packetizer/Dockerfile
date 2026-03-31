# SPDX-FileCopyrightText: 2025-2026 Voyager Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0
ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}
FROM ${BASE_IMAGE} AS base

RUN apt-get update && \
    apt-get install -y rustc cargo && \
    apt-get clean
