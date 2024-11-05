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

FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/clara-holoscan/holoscan:v2.0.0-dgpu

ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.01-0.3.3.1
ARG CACHEBUST=1
ARG BUILDARCH
RUN echo "Using build architecture ${BUILDARCH}"
RUN echo "Using architecture $TARGETPLATFORM"

RUN apt update && apt install -y python3-pyelftools ninja-build meson libyaml-cpp-dev
RUN pip3 install scipy loguru attrs
WORKDIR /opt

ARG DEBIAN_FRONTEND=noninteractive
ARG DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/2.8.0/ubuntu22.04/x86_64

RUN apt update -y \
    && apt install -y --no-install-recommends wget software-properties-common gpg-agent \
    && wget -qO - ${DOCA_REPO_LINK}/GPG-KEY-Mellanox.pub | apt-key add - \
    && add-apt-repository "deb [trusted=yes] ${DOCA_REPO_LINK} ./" \
    && apt update -y \
    && apt install -y --no-install-recommends doca-sdk-aes-gcm doca-sdk-apsh doca-sdk-argp doca-sdk-comch doca-sdk-comm-channel doca-sdk-common doca-sdk-compress doca-sdk-devemu doca-sdk-dma doca-sdk-dpa doca-sdk-dpdk-bridge doca-sdk-erasure-coding doca-sdk-eth doca-sdk-flow doca-sdk-pcc doca-sdk-rdma doca-sdk-sha doca-sdk-telemetry-exporter doca-sdk-urom doca-apsh-config doca-bench doca-caps doca-comm-channel-admin doca-pcc-counters doca-sha-offload-engine doca-socket-relay doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev rdma-core flexio libyara8 \
    && rm -rf /usr/lib/python3/dist-packages \
    && apt remove -y wget software-properties-common gpg-agent \
    && rm -rf /var/lib/apt/lists/* \
    && apt autoremove -y
