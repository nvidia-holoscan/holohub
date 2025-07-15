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

FROM ${BASE_IMAGE} as base-img
ARG DEBIAN_FRONTEND=noninteractive

ENV WORKDIR=/opt/nvidia
WORKDIR ${WORKDIR}

############################################################
# Python environment
############################################################

# Install additional package requirements
RUN apt-get update && \
    apt-get install -y \
        p7zip-full \
        p7zip-rar \
        python3-distutils \
        libjpeg-turbo8-dev
        
# Install Python MONAI Deploy requirements while using the `holoscan` package provided in the base container
COPY ./tutorials/dicom_to_usd_with_monai_and_holoscan/requirements.txt ${WORKDIR}
RUN pip install -q -r requirements.txt && \
    pip uninstall -y holoscan

############################################################
# Pipeline data
############################################################
ARG DATA_NAME="ai_spleen_seg_bundle_data"

# get spleen segmentation model and data
RUN pip install gdown && \
    gdown "https://drive.google.com/uc?id=1Uds8mEvdGNYUuvFpTtCQ8gNU97bAPCaQ&authuser=0&confirm=t&uuid=dc3d74c3-6d43-4f3c-8f2d-613562b8710c&at=APZUnTUrsMMGyOCl-Ik5KxwLyZRZ%3A1707201858378" && \
    unzip -o "${DATA_NAME}.zip" -d "${DATA_NAME}"

############################################################
# OpenUSD libraries
############################################################

# OpenUSD x86_64 wheels are available on PyPI
RUN if [ $(uname -m) = "x86_64" ]; then \
    pip install usd-core; \
fi

# OpenUSD aarch64 Python libraries are available from NVIDIA Omniverse
ENV USD_LIB_DIR="/_build"
RUN if [ $(uname -m) = "aarch64" ]; then \
    mkdir -p ${USD_LIB_DIR}/bindings-python && \
    mkdir -p ${USD_LIB_DIR}/bindings-python/omni && \
    mkdir -p ${USD_LIB_DIR}/python && \
    mkdir -p ${USD_LIB_DIR}/usd; \
fi

RUN if [ $(uname -m) = "aarch64" ]; then \
    wget https://d4i3qtqj3r0z5.cloudfront.net/nv-usd@22.11.nv.0.2.1140.47bb672c-linux-aarch64_py310_release-dev_omniverse.7z && \
        7z x -y nv-usd@22.11.nv.0.2.1140.47bb672c-linux-aarch64_py310_release-dev_omniverse.7z \
            -o${USD_LIB_DIR}/nv-usd && \
    mv ${USD_LIB_DIR}/nv-usd/include/pxr ${USD_LIB_DIR}/python && \
    chmod -R a+rx ${USD_LIB_DIR}/python/ ${USD_LIB_DIR}/nv-usd; \
fi

############################################################
# Container environment
############################################################
    
# MONAI Deploy environment variables
ENV HOLOSCAN_INPUT_PATH ${WORKDIR}/${DATA_NAME}/dcm
ENV HOLOSCAN_MODEL_PATH ${WORKDIR}/${DATA_NAME}/model.ts
ENV HOLOSCAN_OUTPUT_PATH /workspace/holohub/tutorials/dicom_to_usd_with_monai_and_holoscan/output

# Omniverse environment variables
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${USD_LIB_DIR}:${USD_LIB_DIR}/nv-usd/lib
ENV PYTHONPATH=${PYTHONPATH}:/workspace/holohub:${USD_LIB_DIR}/python:${USD_LIB_DIR}/nv-usd/lib/python
