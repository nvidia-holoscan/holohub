# SPDX-FileCopyrightText: 2025-2026 Voyager Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.0.0-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

RUN Cuda_Version=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p' | sed 's/\./-/g') && \
    echo "Version:"$Cuda_Version && \
    test -n "${Cuda_Version}" || (echo "Failed to parse CUDA version from nvcc output" >&2; exit 1) && \
    apt-get update && \
    apt-get install -y \
        rustc \
        cargo \
        python3-numpy \
        libcublas-dev-${Cuda_Version} \
        libcusparse-dev-${Cuda_Version} \
        libcurand-dev-${Cuda_Version} \
        libcufft-dev-${Cuda_Version} \
        libcusolver-dev-${Cuda_Version} && \
    apt-get clean
