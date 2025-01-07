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

# --------------------------------------------------------------------------
#
# Build stage to export the pytorch model to ONNX
#

FROM ${BASE_IMAGE} AS builder

ARG MODEL=vitb

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the Depth-Anything-V2 repository
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git \
    && git clone --depth 1 https://github.com/spacewalk01/depth-anything-tensorrt.git \
    && mv depth-anything-tensorrt/depth_anything_v2/dpt.py Depth-Anything-V2/depth_anything_v2 \
    && mv depth-anything-tensorrt/depth_anything_v2/export_v2.py Depth-Anything-V2

RUN pip install onnx \
    && pip install -r Depth-Anything-V2/requirements.txt

# Change to the repository directory
WORKDIR /app/Depth-Anything-V2

RUN mkdir -p checkpoints \
    && wget -O checkpoints/depth_anything_v2_vitb.pth \
       https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_${MODEL}.pth?download=true

RUN python export_v2.py --encoder vitb --input-size 518


# --------------------------------------------------------------------------
#
# Final stage
#

FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install only minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app/Depth-Anything-V2

# Copy necessary files from the builder stage
COPY --from=builder /app/Depth-Anything-V2/*.onnx /models/.


#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc
