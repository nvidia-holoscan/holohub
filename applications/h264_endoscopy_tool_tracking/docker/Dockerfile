# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Dockerfile to build a docker image on top of Holoscan SDK required to run
# Holohub H264 Encode / Decode applications.

ARG HOLOSCAN_SDK_IMAGE=nvcr.io/nvstaging/holoscan/holoscan
ARG HOLOSCAN_SDK_VERSION=v1.0.0.3
ARG CONTAINER_TYPE=dgpu
FROM ${HOLOSCAN_SDK_IMAGE}:${HOLOSCAN_SDK_VERSION}-${CONTAINER_TYPE}

# Install libv4l-dev required for nvv4l2
RUN apt update && apt install -y libv4l-dev

# Below workaround is required to get H264 Encode working
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

COPY ./docker/install_dependencies.sh /

WORKDIR /

RUN /install_dependencies.sh

