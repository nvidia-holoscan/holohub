# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu as base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub dev setup 
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# - Install libv4l-dev required for nvv4l2
# - Install kmod as a workaround to fix iGPU support.
#   GXF ENC / DEC needs lsmod to check whether it's dGPU or iGPU.
RUN apt update && apt install -y libv4l-dev kmod

# Below workarounds are required to get H.264 Encode and Decode working inside
# the docker container.
RUN if [ ! -e "/usr/lib/$(arch)-linux-gnu/libnvidia-encode.so" ]; then \
        ln -s /usr/lib/$(arch)-linux-gnu/libnvidia-encode.so.1 /usr/lib/$(arch)-linux-gnu/libnvidia-encode.so; \
    fi
RUN mkdir /usr/lib/$(arch)-linux-gnu/libv4l/plugins/nv && \
    ln -s /usr/lib/$(arch)-linux-gnu/tegra/libv4l2_nvcuvidvideocodec.so /usr/lib/$(arch)-linux-gnu/libv4l/plugins/nv/libv4l2_nvcuvidvideocodec.so

COPY applications/h264/install_dependencies.sh /

WORKDIR /

# Uncomment the following line for aarch64 support
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/

RUN /install_dependencies.sh
