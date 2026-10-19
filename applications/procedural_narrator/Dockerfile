# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

ARG GPU_TYPE
ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${BASE_SDK_VERSION}-${GPU_TYPE}

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY applications/procedural_narrator/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Pytest is development-container tooling used by the CTest target.
RUN python3 -m pip install --no-cache-dir "pytest>=8,<10"

RUN mkdir -p /tmp/scripts/utilities
COPY holohub /tmp/scripts/
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub \
    && /tmp/scripts/holohub setup \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/holohub
