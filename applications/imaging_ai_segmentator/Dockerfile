# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG DEBIAN_FRONTEND=noninteractive

FROM ${BASE_IMAGE} AS base

# --------------------------------------------------------------------------
# Holohub run setup
# --------------------------------------------------------------------------
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup && rm -rf /var/lib/apt/lists/*
# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# --------------------------------------------------------------------------
# Benchmarking setup
# --------------------------------------------------------------------------
RUN apt update \
    && apt install --no-install-recommends -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    libgtk-3-dev \
    libcanberra-gtk-module \
    graphviz\
    ninja-build

RUN pip install meson

RUN if ! grep -q "VERSION_ID=\"22.04\"" /etc/os-release; then \
        pip install setuptools; \
    fi
COPY benchmarks/holoscan_flow_benchmarking/requirements.txt /tmp/benchmarking_requirements.txt
RUN pip install -r /tmp/benchmarking_requirements.txt

# --------------------------------------------------------------------------
# Application specific setup
# --------------------------------------------------------------------------
# Environment variables
ENV HOLOSCAN_INPUT_PATH=/workspace/holohub/data/imaging_ai_segmentator/dicom
ENV HOLOSCAN_MODEL_PATH=/workspace/holohub/data/imaging_ai_segmentator/models
ENV HOLOSCAN_OUTPUT_PATH=/workspace/holohub/build/imaging_ai_segmentator/output

# Python dependencies (remove holoscan from requirements.txt)
COPY applications/imaging_ai_segmentator/requirements.txt /tmp/requirements.txt
RUN sed -i '/holoscan/d' /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
