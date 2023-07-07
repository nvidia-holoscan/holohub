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

ARG BASE_IMAGE=ngc_dgpu
# Holoscan SDK 0.5.1 dGPU container from NGC (x86, Clara AGX and IGX Orin Dev Kits)
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.5.1-dgpu AS ngc_dgpu
# Holoscan SDK 0.5.1 ARM64 iGPU container from NGC (AGX Orin and IGX Orin (iGPU mode) Dev Kits)
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.5.1-igpu  AS ngc_igpu
# Holoscan SDK container built from source
FROM holoscan-sdk-dev:latest as local

FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \ 
    apt install --no-install-recommends -y \
    ffmpeg=7:4.2.7-0ubuntu0.1 \
    libv4l-dev=1.18.0-2build1

# --------------------------------------------------------------------------
#
# Holohub run setup 
#

RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup


# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release


