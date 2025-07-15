#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This is the script to run the docker release container.
IMAGE_TAG=holohub:volume_rendering_xr_rel

container_runtime_version=$(nvidia-container-toolkit --version | grep -Po '(?<=version )\d.\d.')
if [[ $(echo -e "$container_runtime_version\n1.2" | sort -V | head -n1) != "1.2" ]]; then
    # at least version 1.2 is needed for Vulkan and EGL support without the need for manually mapping files into the container
    echo "At least NVIDIA Container Runtime version 1.2 is required, but found $container_runtime_version"
    echo "Please visit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html to update."
    exit 1
fi

# Find the nvoptix.bin file
if [ -f "/usr/share/nvidia/nvoptix.bin" ]; then
    mount_nvoptix_bin="-v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro"
fi

# Parse arguments
ARGS=("$@")
SKIP_NEXT=0
for i in "${!ARGS[@]}"; do
    arg="${ARGS[i]}"
    if [ "$SKIP_NEXT" == "1" ]; then
        SKIP_NEXT=0
        continue
    elif [ "$arg" = "--help" ]; then
        echo "Holoscan OpenXR demo docker run script"
        echo ""
        echo "Arguments:"
        echo "  --help : print this help message"
        echo "  -v : pass to docker command to mount volumes (can be specified multiple times)"
        echo ""
        echo "Remaining arguments are passed to the release container entrypoint. Additional options:"
        echo ""
        # pass to docker entry point to print
        arguments="--help"
        break
    elif [ "$arg" = "-v" ]; then
        volumes+="${ARGS[i+1]}"
        SKIP_NEXT="1"
    else
        arguments+=" $arg"
    fi
done

# DOCKER PARAMETERS
#
# -it
#   The container needs to be interactive to be able to interact with the X11 windows
#
# --rm
#   Deletes the container after the command runs
#
# --runtime=nvidia \
#   Enable GPU acceleration
#
# -v /tmp/.X11-unix:/tmp/.X11-unix
# -e DISPLAY
#   Enable graphical applications
#
# --cap-add=sys_nice
#   Allow to change scheduling priorities, required by windrunner-service
#
# ${mount_nvoptix_bin}
#   Mount Optix denoiser weights when present.
#
# ${volumes}
#   User specified volumes to mount.
#
# -v /tmp:/shader_cache
# -e CUDA_CACHE_PATH="/shader_cache/ComputeCache"
# -e OPTIX_CACHE_PATH="/shader_cache/.cache/OptixCache"
# -e __GL_SHADER_DISK_CACHE_PATH="/shader_cache//.cache/GLCache"
#   Persistent shader disk cache
#
# -e STARTED_FROM_SCRIPT=1
#   Indicates to entry point that the container had been started by a script
#
docker run --rm -it \
    --net host \
    -u $(id -u):$(id -g) \
    -e HOME=/tmp \
    --runtime=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    --cap-add=sys_nice \
    ${mount_nvoptix_bin} \
    ${volumes} \
    -v /tmp/.cache:/tmp/.cache \
    -e CUDA_CACHE_PATH="/tmp/.cache/ComputeCache" \
    -e OPTIX_CACHE_PATH="/tmp/.cache/OptixCache" \
    -e __GL_SHADER_DISK_CACHE_PATH="/tmp/.cache/GLCache" \
    -e STARTED_FROM_SCRIPT=1 \
    ${IMAGE_TAG} $arguments
