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


############################################################
# Base image
############################################################


ARG BASE_IMAGE
ARG GPU_TYPE

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-dgpu AS base

ARG DEBIAN_FRONTEND=noninteractive


# --------------------------------------------------------------------------
#
# Install Application Dependencies
#
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install torch torchvision
RUN apt-get update && apt-get install -y wget unzip
RUN if ! command -v ngc > /dev/null; then \
        echo "Installing ngc cli" && \
        ARCH=$(uname -m) && \
        if [ "$ARCH" = "aarch64" ]; then \
            wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_arm64.zip -O ngccli.zip; \
        elif [ "$ARCH" = "x86_64" ]; then \
            wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_linux.zip -O ngccli.zip; \
        fi && \
        unzip ngccli.zip && \
        chmod u+x ngc-cli/ngc && \
        ln -s $(pwd)/ngc-cli/ngc /usr/local/bin/ && \
        rm ngccli.zip; \
    fi
