#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo $SCRIPT_DIR
HOLOSAR_DIR=$SCRIPT_DIR/..
echo $HOLOSAR_DIR

docker run -it \
           --net=host \
           -u $(id -u):$(id -g) \
           -v /etc/group:/etc/group:ro \
           -v /etc/passwd:/etc/passwd:ro \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $HOLOSAR_DIR:/home/docker/holosar \
           -w /home/docker/holosar \
           --device=/dev/bus/usb \
           --runtime=nvidia \
           -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
           -e DISPLAY=$DISPLAY \
           --device=/dev/snd \
           holosar_app:latest \
           python holosar.py 
