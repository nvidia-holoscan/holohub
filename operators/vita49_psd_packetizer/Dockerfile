# SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v2.9.0-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

RUN apt-get update && \
    apt-get install -y rustc cargo && \
    apt-get clean
