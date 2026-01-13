# syntax=docker/dockerfile:1

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


############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Install PortAudio to enable PyAudio
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy over app-specific requirements
COPY applications/asr_to_llm/requirements.txt /tmp/requirements.txt

# Install setuptools prior to all other requirements to avoid install errors
RUN python3 -m pip install setuptools \
    && CMAKE_ARGS="-DLLAMA_CUBLAS=on"  \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
