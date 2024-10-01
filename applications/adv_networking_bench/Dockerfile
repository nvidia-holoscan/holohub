# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} as base

ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.07-0.6.1.0
ARG CACHEBUST=1
ARG BUILDARCH
RUN echo "Using build architecture ${BUILDARCH}"
RUN echo "Using architecture $TARGETPLATFORM"

RUN apt update && apt install -y python3-pyelftools ninja-build meson
RUN pip3 install scipy loguru attrs
WORKDIR /opt

RUN    eval `dpkg-architecture` \
        && wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.7.0/host/doca-host_2.7.0-204000-24.04-ubuntu2204_${BUILDARCH}.deb -O doca-host.deb \
        && ls -lh && apt install ./doca-host.deb \
        && apt update \
        && apt install -y doca-all \
        && apt install -y doca-gpu doca-gpu-dev
