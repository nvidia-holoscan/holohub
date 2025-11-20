# syntax=docker/dockerfile:1
#
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

############################################################
# Surgical Scene Reconstruction - Production Dockerfile v2.0
#
# This Dockerfile creates a production-ready container for
# real-time surgical scene reconstruction using Gaussian
# Splatting with temporal deformation in Holoscan.
#
# Features:
#   - Dynamic rendering with temporal deformation
#   - Real-time Holoviz visualization
#   - 3D Gaussian Splatting (gsplat)
#   - EndoNeRF dataset support
#   - Tool removal capability
#   - On-the-fly training support (NEW in v2.0)
############################################################

# Use configurable base image (default: NGC Holoscan v3.7.0)
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.7.0-cuda12-dgpu
FROM ${BASE_IMAGE} AS base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

############################################################
# Install System Dependencies
############################################################

# Install OpenGL libraries required by open3d
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

############################################################
# Install Python Dependencies
############################################################

# Install all dependencies from training requirements file
# Includes both inference and training dependencies with version specifications
# All versions tested and confirmed working in production
COPY applications/surgical_scene_recon/training/training_requirements.txt /tmp/training_requirements.txt
RUN pip install --no-cache-dir -r /tmp/training_requirements.txt

############################################################
# Environment Configuration
############################################################

# Set Python path for Holoscan SDK
ENV PYTHONPATH="/opt/nvidia/holoscan/python/lib:${PYTHONPATH}"

# Set working directory
WORKDIR /workspace/holohub/applications/surgical_scene_recon

############################################################
# Container Metadata
############################################################

LABEL maintainer="NVIDIA Holoscan Team"
LABEL description="Surgical Scene Reconstruction with 3D Gaussian Splatting and Temporal Deformation"
LABEL version="2.0-production-training"
LABEL holoscan.sdk.version="3.7.0"
LABEL application="surgical_scene_recon"
LABEL rendering.modes="static,dynamic"
LABEL operation.modes="inference,training"
LABEL dataset.support="EndoNeRF"
LABEL updated="2025-10-24"

############################################################
# Default Command
############################################################

# Show usage information by default
CMD ["bash", "-c", "echo 'Surgical Scene Reconstruction Container' && \
     echo '' && \
     echo 'Available test scripts:' && \
    echo '  - demo_dynamic_rendering_viz.py (high-quality, temporal deformation)' && \
    echo '  - demo_static_rendering_viz.py (fast preview mode)' && \
    echo '' && \
    echo 'Example usage:' && \
    echo '  python demo_dynamic_rendering_viz.py --data_dir /path/to/data --checkpoint /path/to/checkpoint.pt' && \
     echo '' && \
     echo 'See README.md and QUICKSTART.md for detailed instructions.'"]
