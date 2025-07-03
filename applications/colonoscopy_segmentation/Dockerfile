# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvcr.io/nvidia/cuda:12.9.0-base-ubuntu24.04

# Install the dependencies
# wget, unzip and ffmpeg are needed to download NGC-CLI and generate input data
# python packages are required for the application
# libnpp, cuda-nvrtc and libnvinfer are required for Holoscan inference
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        unzip \
        cmake \
        make \
        patch \
        ffmpeg \
        libnpp-12-9 \
        cuda-nvrtc-12-9 \
        libnvinfer10 \
        libnvinfer-plugin10 \
        libnvonnxparsers10 \
        && rm -rf /var/lib/apt/lists/*

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_${ARCH}.zip -O ngccli_linux.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && chmod u+x ngc-cli/ngc \
    && ln -s $(pwd)/ngc-cli/ngc /usr/local/bin/

# Install python3 if not present (needed for holohub CLI)
ARG PYTHON_VERSION=python3
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install --no-install-recommends -y \
            software-properties-common curl gpg-agent \
        && add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install --no-install-recommends -y \
            ${PYTHON_VERSION} \
        && apt purge -y \
            python3-pip \
            software-properties-common \
        && apt-get autoremove --purge -y \
        && rm -rf /var/lib/apt/lists/* \
        && update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON_VERSION} 100 \
        && if [ "${PYTHON_VERSION}" != "python3" ]; then \
            update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON_VERSION} 100 \
            ; fi \
    ; fi
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if ! python3 -m pip --version >/dev/null 2>&1; then \
        curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON_VERSION} \
    ; fi

RUN pip3 install holoscan cupy-cuda12x \
    && chmod -R a+w /usr/local/lib/python3.12/dist-packages/holoscan-3.4.0.dist-info/ \
    && chmod -R a+rw /usr/local/lib/python3.12/dist-packages/holoscan
