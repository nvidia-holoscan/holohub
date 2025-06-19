# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


FROM nvcr.io/nvidia/cuda:12.9.0-base-ubuntu24.04

# Install the dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        cmake \
        make \
        patch \
        ffmpeg \
        python3 \
        python3-venv \
        libnpp-12-9 \
        cuda-nvrtc-12-9 \
        libnvinfer10 \
        libnvinfer-plugin10 \
        libnvonnxparsers10 \
        && rm -rf /var/lib/apt/lists/*

RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && curl -S -# -L -o ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_${ARCH}.zip \
    && unzip -q ngccli_linux.zip \
    && rm ngccli_linux.zip \
    && chmod u+x ngc-cli/ngc \
    && ln -s $(pwd)/ngc-cli/ngc /usr/local/bin/

RUN python3 -m venv /local-venv
RUN /local-venv/bin/pip3 install holoscan
RUN /local-venv/bin/pip3 install cupy-cuda12x

ENV PATH="/local-venv/bin:$PATH"