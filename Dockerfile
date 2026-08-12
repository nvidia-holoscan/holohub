# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# BuildKit provides TARGETARCH (amd64 or arm64), but the SDK's local image
# names use x86_64 or aarch64. The holohub wrapper performs that explicit
# mapping and supplies the matching image here.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install and verify the wrapper-pinned Holoscan CLI. Application-specific
# images add their own build and runtime dependencies.
COPY --chmod=755 holohub /tmp/scripts/
RUN /tmp/scripts/holohub env-info
