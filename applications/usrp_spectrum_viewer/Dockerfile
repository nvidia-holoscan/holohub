# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
ARG NETWORKING_BASE_IMAGE=holohub-networking:4.5.0
FROM ${NETWORKING_BASE_IMAGE}

ARG UHD_GIT_REF=v4.10.0.0

# UHD and its build prerequisites.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
        pybind11-dev \
        python3-dev \
        python3-mako \
        python3-numpy \
        python3-requests \
        python3-ruamel.yaml \
    && git clone --branch "${UHD_GIT_REF}" --depth 1 \
        https://github.com/EttusResearch/uhd.git /tmp/uhd \
    && cmake -S /tmp/uhd/host -B /tmp/uhd/host/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TESTS=OFF \
        -DENABLE_MANUAL=OFF \
    && cmake --build /tmp/uhd/host/build -j"$(nproc)" \
    && cmake --install /tmp/uhd/host/build \
    && ldconfig \
    && rm -rf /tmp/uhd /var/lib/apt/lists/*
