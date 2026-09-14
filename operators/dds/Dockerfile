# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_SDK_VERSION=4.5.0
ARG RTI_CONNEXT_VERSION=7.7.0

# Install the RTI Connext SDK in the image so CMake can find the native
# libraries and rtiddsgen without requiring a host installation.
RUN rm -f /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates curl openjdk-21-jre v4l-utils xauth xvfb \
    && curl -sSL -o /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official/repo.key \
    && printf -- "deb [arch=%s signed-by=%s] %s %s main\n" \
        "$(dpkg --print-architecture)" \
        /usr/share/keyrings/rti-official-archive.gpg \
        https://packages.rti.com/deb/official \
        "$(. /etc/os-release && echo "${VERSION_CODENAME}")" \
        > /etc/apt/sources.list.d/rti.list \
    && RTI_LICENSE_AGREEMENT_ACCEPTED=accepted apt-get update \
    && RTI_LICENSE_AGREEMENT_ACCEPTED=accepted apt-get install --no-install-recommends -y \
        "rti-connext-dds-${RTI_CONNEXT_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# HoloHub starts non-interactive Bash shells inside the image. BASH_ENV selects
# the native Connext target before CMake searches for its libraries.
COPY --chmod=0644 operators/dds/connext_env.sh /opt/rti.com/connext_env.sh
ENV BASH_ENV=/opt/rti.com/connext_env.sh

# HoloHub invokes its build and test commands through holoscan-cli inside the
# image. Keeping the CLI in the image makes this Dockerfile self-contained.
RUN python3 -m pip install --no-cache-dir "holoscan-cli==${BASE_SDK_VERSION}"

ENV NDDSHOME=/opt/rti.com/rti_connext_dds-${RTI_CONNEXT_VERSION}
ENV RTI_CONNEXT_DDS_DIR=${NDDSHOME}
ENV RTI_LICENSE_FILE=${NDDSHOME}/rti_license.dat
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data

CMD ["/bin/bash"]
