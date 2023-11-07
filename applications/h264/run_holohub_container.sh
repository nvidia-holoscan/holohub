#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script starts holohub dev container required to run
# Holohub H264 Encode / Decode applications.

IMAGE=holohub_dev_container
SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE"`
VERSION=`cat $ROOT/VERSION`
HOLOHUB_ROOT=`realpath "$HERE/../../"`

xhost +local:docker

# Find the nvidia_icd.json file which could reside at different paths
# Needed due to https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f,l -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)

# --ipc=host, --cap-add=CAP_SYS_PTRACE, --ulimit memlock=-1 are needed for the distributed applications using UCX to work.
# See https://openucx.readthedocs.io/en/master/running.html#running-in-docker-containers

# Run the container
docker run -it --rm --net host \
  --runtime=nvidia \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $nvidia_icd_json:$nvidia_icd_json:ro \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -v $PWD:$PWD \
  -v $HOLOHUB_ROOT:$HOLOHUB_ROOT \
  -w $HOLOHUB_ROOT \
  -e DISPLAY=$DISPLAY \
  --ipc=host \
  --cap-add=CAP_SYS_PTRACE \
  --ulimit memlock=-1 \
  $IMAGE:$VERSION
