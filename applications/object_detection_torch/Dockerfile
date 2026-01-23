# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Set up Holoscan SDK container libtorch to be found with ldconfig for app C++ build and runtime
# Verify PyTorch is available in base image
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} found')"

# Set up Holoscan SDK container libtorch to be found with ldconfig for app C++ build and runtime
RUN echo $(python3 -c "import torch; print(torch.__path__[0])")/lib > /etc/ld.so.conf.d/libtorch.conf \
    && ldconfig \
    && ldconfig -p | grep -q "libtorch.so"

ARG GPU_TYPE
ARG CUDA_MAJOR
RUN INDEX_URL=""; \
    if [ "${GPU_TYPE}" = "igpu" ]; then \
        INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"; \
    elif [ "${CUDA_MAJOR}" = "12" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cu129"; \
    elif [ "${CUDA_MAJOR}" = "13" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cu130"; \
    fi \
    && python3 -m pip install torchvision --no-cache-dir --index-url ${INDEX_URL}
