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

# Options for setup script
ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="true"
ARG USERNAME=holoscan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

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

# Install Holoscan CLI dependencies
RUN pip install $(sed -n '/done <<EOF/,/EOF/{/done <<EOF/b;/EOF/b;p}' /opt/nvidia/holoscan/bin/holoscan)

# Set the default user.
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
USER $USERNAME
