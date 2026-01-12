# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu

ARG HOLOHUB_DIR=/opt/holohub-internal

# Architecture setup
# "x86_64" OR "aarch64"
ARG ARCH=x86_64
# "arm64" OR "amd64"
ARG DEB_ARCH=amd64

# Version parameters
ARG OFED_VERSION=5.8-1.0.1.1
ARG UBUNTU_VERSION=20.04
ARG DPDK_VERSION=22.11.1
ARG NATS_VER=v2.9.8
ARG NATS_FILE=nats-server-${NATS_VER}-${DEB_ARCH}.deb

# Mellanox OFED
ARG MLNX_FILE=MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-${ARCH}
RUN wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-${OFED_VERSION}/${MLNX_FILE}.tgz && \
        tar xvf ${MLNX_FILE}.tgz && \
        cd ${MLNX_FILE} && \
        ./mlnxofedinstall --upstream-libs --dpdk --with-mft --upstream-libs --user-space-only --force --without-fw-update && \
        cd ../ && \
        rm -fr ${MLNX_FILE} && \
        rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y python3-pyelftools ninja-build meson libbsd-dev sudo

# Download and unzip DPDK
ADD https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz /tmp/
RUN cd /tmp && tar xf dpdk-${DPDK_VERSION}.tar.xz

# Setup DPDK and apply patches from advanced network operator
ARG ADV_NET_DIR=operators/advanced_network/
COPY ${ADV_NET_DIR}/dpdk_patches/*.patch /tmp/dpdk-stable-${DPDK_VERSION}

WORKDIR /tmp/dpdk-stable-${DPDK_VERSION}/
RUN patch --ignore-whitespace --fuzz 3 config/arm/meson.build /tmp/dpdk-stable-${DPDK_VERSION}/dpdk.nvidia.patch
RUN patch --ignore-whitespace --fuzz 3 drivers/gpu/cuda/devices.h /tmp/dpdk-stable-${DPDK_VERSION}/devices.h.patch
RUN patch --ignore-whitespace --fuzz 3 drivers/gpu/cuda/cuda.c /tmp/dpdk-stable-${DPDK_VERSION}/cuda.c.patch
RUN CFLAGS=-I/usr/local/cuda/include meson build -Dplatform=generic -Dc_args=-I/usr/local/cuda/include \
          -Ddisabled_drivers=baseband/*,bus/ifpga/*,common/cpt,common/dpaax,common/iavf,common/octeontx,common/octeontx2,crypto/nitrox,net/ark,net/atlantic,net/avp,net/axgbe,net/bnx2x,net/bnxt,net/cxgbe,net/e1000,net/ena,net/enic,net/fm10k,net/hinic,net/hns3,net/i40e,net/ixgbe,vdpa/ifc,net/igc,net/liquidio,net/netvsc,net/nfp,net/qede,net/sfc,net/thunderx,net/vdev_netvsc,net/vmxnet3,regex/octeontx2,
RUN ninja -C build install
RUN rm /tmp/dpdk-${DPDK_VERSION}.tar.xz

WORKDIR /tmp/
RUN wget https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/${NATS_FILE} && \
        dpkg -i ${NATS_FILE}
RUN rm /tmp/${NATS_FILE}

RUN pip3 install scipy plotly plotly_express matplotlib dash dash_bootstrap_components nats-py loguru attrs