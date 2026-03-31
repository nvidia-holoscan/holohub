# SPDX-FileCopyrightText: 2025-2026 Voyager Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0
ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

RUN CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p' | sed 's/\./-/g') && \
    echo "Version:"$CUDA_VERSION && \
    test -n "${CUDA_VERSION}" || (echo "Failed to parse CUDA version from nvcc output" >&2; exit 1) && \
    apt-get update && \
    apt-get install -y \
        rustc \
        cargo \
        python3-numpy \
        libcublas-dev-${CUDA_VERSION} \
        libcusparse-dev-${CUDA_VERSION} \
        libcurand-dev-${CUDA_VERSION} \
        libcufft-dev-${CUDA_VERSION} \
        libcusolver-dev-${CUDA_VERSION} && \
    apt-get clean
