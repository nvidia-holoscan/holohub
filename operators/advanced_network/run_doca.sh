#!/bin/sh

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
#
# See README.md for detailed information.

set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath $HERE/..`

nvidia_icd_json=$(find -L /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)
docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --privileged \
    --name demo \
    -v $PWD:$PWD \
    -v $PWD/..:$PWD/.. \
    -v /sys/bus/pci/devices:/sys/bus/pci/devices \
    -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -w $PWD \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    docker.io/library/holohub-doca:doca-28-ubuntu2204 \
    $*

# To build Docker image: ./dev_container build --docker_file operators/advanced_network/Dockerfile --img holohub-doca:doca-27-ubuntu2204 --no-cache
# Launch DOCA container ./operators/advanced_network/run_doca.sh
# To build operator + app from main dir: ./run build adv_networking_bench
# Run rx only: ./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_rx.yaml
