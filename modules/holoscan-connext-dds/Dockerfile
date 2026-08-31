# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_SDK_VERSION=4.5.0
ARG RTI_CONNEXT_VERSION=7.7.0

# Configure RTI's official APT repository and install the Connext SDK. The SDK
# provides rtiddsgen and the native libraries needed by generated DDS types.
RUN rm -f /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl openjdk-21-jre xvfb \
    && curl -sSL -o /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official/repo.key \
    && printf -- "deb [arch=%s signed-by=%s] %s %s main\n" \
        "$(dpkg --print-architecture)" \
        /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official \
        "$(. /etc/os-release && echo "${VERSION_CODENAME}")" \
        > /etc/apt/sources.list.d/rti.list \
    && RTI_LICENSE_AGREEMENT_ACCEPTED=accepted apt-get update \
    && RTI_LICENSE_AGREEMENT_ACCEPTED=accepted apt-get install -y \
        "rti-connext-dds-${RTI_CONNEXT_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# The Debian SDK does not install the Python API used by these operators.
# Pytest is included so the operator can be validated entirely in the container.
RUN python3 -m pip install --no-cache-dir \
    "rti.connext==${RTI_CONNEXT_VERSION}" \
    "holoscan-cli==${BASE_SDK_VERSION}" \
    pytest \
    build

ENV NDDSHOME=/opt/rti.com/rti_connext_dds-${RTI_CONNEXT_VERSION}
ENV RTI_CONNEXT_DDS_DIR=${NDDSHOME}
ENV RTI_LICENSE_FILE=${NDDSHOME}/rti_license.dat
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

CMD ["/bin/bash"]
