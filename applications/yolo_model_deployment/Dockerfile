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

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

FROM base AS holohub-setup

# --------------------------------------------------------------------------
#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
#
# Install required apt packages and python modules
#
FROM holohub-setup AS holohub-yolo

RUN apt-get update \
    && apt-get install -y libgl1

RUN pip3 install \
    ultralytics \
    onnx \
    onnx-graphsurgeon

ARG GPU_TYPE
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN if [ "${GPU_TYPE}" == "igpu" ]; then \
        python3 -m pip list | grep -i torch | awk '{print $1}' | xargs -r python3 -m pip uninstall -y; \
        python3 -m pip install torch torchvision torchaudio --force-reinstall --index-url "https://pypi.jetson-ai-lab.io/jp6/cu129"; \
    fi

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc
