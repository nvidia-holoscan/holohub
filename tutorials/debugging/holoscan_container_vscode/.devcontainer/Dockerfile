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

ARG HOLOSCAN_SDK_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v2.2.0-dgpu
FROM ${HOLOSCAN_SDK_IMAGE}

ARG WORKSPACE_DIR=/workspace

ARG USERNAME=holoscan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Remove existing conflicting user
# see https://bugs.launchpad.net/cloud-images/+bug/2005129
RUN getent passwd $USER_UID | cut -d: -f1 | xargs -r userdel -r || true

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME /home/$USERNAME

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# Install additional packages
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends  \
        build-essential gdb \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy VS Code launch profiles and rename README.md
RUN mkdir -p $WORKSPACE_DIR/.vscode \
    && cp /opt/nvidia/holoscan/.vscode/launch.json $WORKSPACE_DIR/.vscode \
    && cp /opt/nvidia/holoscan/.vscode/debug_python $WORKSPACE_DIR/.vscode \
    && chown -R $USERNAME $WORKSPACE_DIR/.vscode \
    && sed -i 's|/opt/nvidia/holoscan/.vscode|/workspace/.vscode|g' $WORKSPACE_DIR/.vscode/* \
    && mv $WORKSPACE_DIR/README.md $WORKSPACE_DIR/TensorRT-README.md 

# Set the default user.
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
USER $USERNAME
