# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG MODEL=vits

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

RUN git clone --depth 1 https://github.com/deepmind/tapnet.git

# Override tapnext modules with custom versions before installing
COPY applications/tracks2endo4d/onnx_conversion/tapnext_torch.py tapnet/tapnet/tapnext/tapnext_torch.py
COPY applications/tracks2endo4d/onnx_conversion/tapnext_lru_modules.py tapnet/tapnet/tapnext/tapnext_lru_modules.py

RUN cd tapnet && pip install .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install onnx onnxruntime numpy einops onnxscript

RUN mkdir -p checkpoints \
    && wget --progress=dot:giga --tries=3 --timeout=300 \
       https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz \
       -O checkpoints/bootstapnext_ckpt.npz

# Copy the ONNX tapnext conversion script to the builder stage and run the conversion
COPY applications/tracks2endo4d/onnx_conversion/onnx_tapnext.py .

RUN python onnx_tapnext.py --checkpoint checkpoints/bootstapnext_ckpt.npz --output_path checkpoints/ --device cpu

# --------------------------------------------------------------------------
#
# Final stage
#

FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install NGC CLI
RUN if [ $(uname -m) = "aarch64" ]; then ARCH=arm64; else ARCH=linux; fi \
    && wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.3/files/ngccli_${ARCH}.zip -O /tmp/ngccli_linux.zip \
    && unzip /tmp/ngccli_linux.zip -d /tmp \
    && chmod u+x /tmp/ngc-cli/ngc \
    && mv /tmp/ngc-cli /usr/local/ngc-cli \
    && ln -s /usr/local/ngc-cli/ngc /usr/local/bin/ngc \
    && rm /tmp/ngccli_linux.zip

# Set the working directory
WORKDIR /app/tracks2endo4d

# Copy necessary files from the builder stage
COPY --from=builder /app/checkpoints/*.onnx* /models/.
