# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS v4l2-depth-base

COPY --chmod=755 holohub /tmp/scripts/
RUN /tmp/scripts/holohub env-info

# Inference builds prepare the model through the application's CMake target
# into the mounted project data directory, keeping artifacts out of image layers.
FROM v4l2-depth-base AS v4l2-depth-live

RUN apt-get update \
    && apt-get install -y --no-install-recommends git wget libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        onnx==1.19.1 \
        onnxscript==0.7.1 \
        opencv-python-headless==5.0.0.93

# Loopback mode runs its file decoder and V4L2 producer in the same container
# as the application. The kernel module and /dev/video42 remain host resources.
FROM v4l2-depth-live AS v4l2-depth-loopback

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gir1.2-gstreamer-1.0 \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-tools \
        python3-gi \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Graph validation does not start operator lifecycles, so it does not need the
# model-export dependencies. Keep this lightweight stage as the default target.
FROM v4l2-depth-base AS v4l2-depth-validate
