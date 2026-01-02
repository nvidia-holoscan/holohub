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
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install curl for downloading data files
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add schemas directory to PYTHONPATH to enable importing operators
ENV PYTHONPATH=$PYTHONPATH:/workspace/holohub

# Install Python dependencies
RUN pip install nibabel nilearn

# Install Kernel's dependencies
RUN pip install \
    numpy \
    scipy \
    h5py