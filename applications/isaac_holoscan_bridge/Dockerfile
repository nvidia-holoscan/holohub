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

ARG HOLOLINK_VERSION=2.0.0

ARG ISAAC_SIM_VERSION=4.5.0
ARG ISAAC_SIM_DOCKER_SERVER=nvcr.io/nvidia/isaac-sim

# Hololink demo image for aarch64
FROM --platform=linux/arm64 hololink-demo:${HOLOLINK_VERSION} AS target_arm64

# no setup required, we use the Hololink demo image as is

# IsaacSim image for x86_64
ARG ISAAC_SIM_VERSION
ARG ISAAC_SIM_DOCKER_SERVER
FROM --platform=linux/amd64 ${ISAAC_SIM_DOCKER_SERVER}:${ISAAC_SIM_VERSION} AS target_amd64

ARG PYTHON_VERSION=3.11

ARG HOLOSCAN_VERSION=3.2.*

# Hololink requires CUDA Forward Compatibility Package, CUDA NVTX, CUDA Runtime, libibverbs1 and DOCA
# Holoscan requires libvulkan1, libegl1 for the Holoviz operator and libnpp for the BayerDemosaic operator
# CuPy needs cuda-nvrtc and cuda-cudart-dev
# Add gdb to the image to allow for debugging
# Remove mesa-vulkan-drivers to workaround https://jirasw.nvidia.com/browse/CLARAHOLOS-1854 (will be fixed in HSDK 3.3)
ARG CUDA_TOOLKIT_VERSION=12.6
ARG CUDA_KEYRING_NAME=cuda-keyring_1.1-1_all.deb
RUN apt update && \
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING_NAME} && \
    dpkg -i ${CUDA_KEYRING_NAME} && \
    apt install -y --no-install-recommends \
        gnupg2 && \
    DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/latest/ubuntu22.04/x86_64 && \
    LOCAL_GPG_KEY_PATH="/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub" && \
    curl ${DOCA_REPO_LINK}/GPG-KEY-Mellanox.pub | gpg --dearmor >${LOCAL_GPG_KEY_PATH} && \
    echo "deb [signed-by=${LOCAL_GPG_KEY_PATH}] ${DOCA_REPO_LINK} ./" | tee /etc/apt/sources.list.d/doca.list && \
    apt update && \
    CUDA_TOOLKIT_VERSION_DASH="$(echo ${CUDA_TOOLKIT_VERSION} | tr '.' '-')" && \
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        cuda-compat-${CUDA_TOOLKIT_VERSION_DASH} \
        cuda-nvtx-${CUDA_TOOLKIT_VERSION_DASH} \
        cuda-cudart-${CUDA_TOOLKIT_VERSION_DASH} \
        libibverbs1 \
        doca-runtime \
        libnpp-${CUDA_TOOLKIT_VERSION_DASH} \
        libvulkan1 \
        cuda-nvrtc-${CUDA_TOOLKIT_VERSION_DASH} \
        cuda-cudart-dev-${CUDA_TOOLKIT_VERSION_DASH} \
        gdb && \
    update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-${CUDA_TOOLKIT_VERSION} 100 && \
    apt remove -y --purge mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/* && \
    rm ${CUDA_KEYRING_NAME}

# install Holoscan Python
RUN ./python.sh -m pip install holoscan==${HOLOSCAN_VERSION}
# The Python wheel of Holoscan sensor bridge links to the shared libraries of the deb package, use the Holoscan Python package instead
RUN echo "/isaac-sim/kit/python/lib/python3.11/site-packages/holoscan/lib" > /etc/ld.so.conf.d/holoscan.conf && \
    ldconfig

# install Holoscan Sensor Bridge
ARG HOLOLINK_WHEEL=hololink-2.0.0-cp311-cp311-linux_x86_64.whl
COPY ${HOLOLINK_WHEEL} /tmp
RUN ./python.sh -m pip install /tmp/${HOLOLINK_WHEEL}

# workaround for https://nvbugspro.nvidia.com/bug/5180229: [Holoscan SDK v3.1] std::bad_alloc exception with IsaacSim due to loading Holoscan Python module with RTDL_GLOBAL flag
RUN sed -i 's/sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)/#sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)/' /isaac-sim/kit/python/lib/python${PYTHON_VERSION}/site-packages/holoscan/core/__init__.py

# IsaacSim is using its own Python installation, so we install alternatives to use that with the normal `python` and `python3` commands.
RUN update-alternatives --install /usr/bin/python python /isaac-sim/python.sh 100 && \
    update-alternatives --install /usr/bin/python3 python3 /isaac-sim/python.sh 100

# update the SCRIPT_DIR to use the realpath which handles symlinks
RUN sed -i 's/SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE\[0\]}" )" && pwd )"/SCRIPT_DIR=$( dirname -- "\$( readlink -f -- "\$0"; )"; )/' /isaac-sim/python.sh

# patch the camera device to return data in CUDA memory
COPY isaac_sim_camera_device.patch /tmp/isaac_sim_camera_device.patch
RUN case "$ISAAC_SIM_VERSION" in \
    *"4.5.0"*) \
        patch -p1 /isaac-sim/exts/isaacsim.sensors.camera/isaacsim/sensors/camera/camera.py </tmp/isaac_sim_camera_device.patch && \
        rm /tmp/isaac_sim_camera_device.patch;; \
    *) \
        ;; \
    esac

FROM target_${TARGETARCH} AS final
