# SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.1.0-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

RUN apt-get update && \
    apt-get install -y rustc cargo && \
    apt-get clean

ARG MATX_VERSION=0.9.0

COPY applications/psd_pipeline/matx_patches /tmp/matx_patches/
RUN mkdir /tmp/matx && \
    cd /tmp/matx && \
    curl -OL "https://github.com/NVIDIA/MatX/archive/refs/tags/v${MATX_VERSION}.tar.gz" && \
    tar -xzf "v${MATX_VERSION}.tar.gz" && \
    cd "./MatX-${MATX_VERSION}" && \
    for p in /tmp/matx_patches/*.patch; do patch -p1 < "$p"; done && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd /tmp && \
    rm -rf /tmp/matx
