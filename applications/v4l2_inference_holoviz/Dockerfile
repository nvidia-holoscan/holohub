# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Supply a development image containing matching Holoscan 5.x and camera installations.
# The repository's default 4.x image does not provide this app's module interfaces.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Bootstrap the same CLI version as the repository wrapper, inside the image.
COPY holohub /tmp/v4l2-app-cli/holohub
RUN bash /tmp/v4l2-app-cli/holohub version
RUN python3 -m pip install --no-cache-dir --break-system-packages 'onnx>=1.17,<2'

WORKDIR /workspace/holohub
