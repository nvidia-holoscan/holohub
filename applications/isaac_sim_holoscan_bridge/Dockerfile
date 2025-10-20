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


ARG ISAAC_SIM_VERSION=4.5.0
ARG ISAAC_SIM_DOCKER_SERVER=nvcr.io/nvidia/isaac-sim

FROM ${ISAAC_SIM_DOCKER_SERVER}:${ISAAC_SIM_VERSION}

ARG PYTHON_VERSION=3.10

ARG HOLOSCAN_VERSION=3.3.*

# Holoscan requires libvulkan1 for the Holoviz operator
# CMake and build-essential are required by Holohub application build
# Add patch to be able to patch Isaac Sim
# Remove mesa-vulkan-drivers to avoid that HolovizOp is trying to use the Vulkan SW renderer (will be fixed in HSDK 3.3)
ARG CUDA_TOOLKIT_VERSION=12.6
ARG CUDA_KEYRING_NAME=cuda-keyring_1.1-1_all.deb
RUN apt update && \
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_NAME} && \
    dpkg -i ${CUDA_KEYRING_NAME} && \
    apt update && \
    CUDA_TOOLKIT_VERSION_DASH="$(echo ${CUDA_TOOLKIT_VERSION} | tr '.' '-')" && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        cuda-compat-${CUDA_TOOLKIT_VERSION_DASH} \
        cuda-cudart-${CUDA_TOOLKIT_VERSION_DASH} \
        libvulkan1 \
        cmake \
        build-essential \
        patch && \
    update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-${CUDA_TOOLKIT_VERSION} 100 && \
    apt remove -y --purge mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/* && \
    rm ${CUDA_KEYRING_NAME}

# install Holoscan Python
RUN ./python.sh -m pip install holoscan==${HOLOSCAN_VERSION}

# Isaac Sim is using its own Python installation, so we install alternatives to use that with the normal `python` and `python3` commands.
RUN update-alternatives --install /usr/bin/python python /isaac-sim/python.sh 100 && \
    update-alternatives --install /usr/bin/python3 python3 /isaac-sim/python.sh 100

# update the SCRIPT_DIR to use the realpath which handles symlinks
RUN sed -i 's/SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE\[0\]}" )" && pwd )"/SCRIPT_DIR=$( dirname -- "\$( readlink -f -- "\$0"; )"; )/' /isaac-sim/python.sh

# patch the camera device to return data in CUDA memory
COPY applications/isaac_sim_holoscan_bridge/isaac_sim_camera_device.patch /isaac_sim_camera_device.patch
RUN patch -p1 /isaac-sim/exts/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py </isaac_sim_camera_device.patch && \
    rm /isaac_sim_camera_device.patch

# The Isaac Sim Image defines an entrypoint to start the simulator in headless mode, so we need to override it
ENTRYPOINT [""]
