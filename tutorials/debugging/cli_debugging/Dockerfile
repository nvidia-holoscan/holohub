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

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS base

# Install Debian packages
# Note: ffmpeg is not required for debugging and is not used in the tutorial.
#   However, some configurations of the Endoscopy Tool Tracking app rely on ffmpeg.
#   This avoids a build error in the case that the Endoscopy Tool Tracking build directory
#   is not clean before the tutorial runs.
RUN apt update && \
    apt install --no-install-recommends -y \
        gdb \
        ffmpeg

ENV PYTHONPATH=/opt/nvidia/holoscan/python/lib:/workspace/holohub/benchmarks/holoscan_flow_benchmarking:/usr/share/gdb/python
