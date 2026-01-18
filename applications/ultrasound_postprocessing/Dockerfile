# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

COPY /applications/ultrasound_postprocessing/pyproject.toml /workspace/holohub/applications/ultrasound_postprocessing/pyproject.toml
RUN python3 -m pip install -e /workspace/holohub/applications/ultrasound_postprocessing[all] \
    && python3 -c "pass";

RUN echo "/workspace/holohub/applications/ultrasound_postprocessing" > \
    /usr/local/lib/python3.12/dist-packages/ultra_post.pth

############################################################
# Test dependencies
############################################################
FROM base as test-setup
RUN apt update && \
    apt install -y --no-install-recommends \
        xvfb
