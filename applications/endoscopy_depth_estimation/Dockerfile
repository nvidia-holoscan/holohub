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
FROM ${BASE_IMAGE} as base

#############################################################
# Install common dependencies via HoloHub CLI
#############################################################
ENV DEBIAN_FRONTEND=noninteractive

FROM base as holohub-setup
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

#############################################################
# Install Runtime Dependencies
#############################################################
FROM holohub-setup as run

RUN apt-get update && apt-get install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm \
    libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libjsoncpp-dev ffmpeg \
     && apt-get clean

# Install OpenCV
ARG OPEN_CV_VERSION=4.10
RUN python3 -m pip install opencv-python~=${OPEN_CV_VERSION}
