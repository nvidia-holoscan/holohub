#!/bin/bash
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
set -e

GIT_ROOT=$(readlink -f ./$(git rev-parse --show-cdup))

# Utilities

YELLOW="\e[1;33m"
RED="\e[1;31m"
NOCOLOR="\e[0m"

print_error() {
    echo -e "${RED}ERROR${NOCOLOR}:" $*
}

get_host_gpu() {
    if ! command -v nvidia-smi >/dev/null; then
        print_error Y "Could not find any GPU drivers on host. Defaulting build to target dGPU/CPU stack."
        echo -n "dgpu"
    elif nvidia-smi  2>/dev/null | grep nvgpu -q; then
        echo -n "igpu"
    else
        echo -n "dgpu"
    fi
}

get_host_arch() {
    echo -n "$(uname -m)"
}

if [ ! -d $GIT_ROOT/build/object_detection_torch ]; then
    print_error "Please build the Object Detection Torch application first with the following command:"
    print_error "./dev_container build_and_run object_detection_torch"
    exit -1
fi

APP_PATH="$GIT_ROOT/object_detection_torch"
echo Creating application directory $APP_PATH...
mkdir -p $APP_PATH
echo Copying application files to $APP_PATH...
cp -f $GIT_ROOT/build/object_detection_torch/applications/object_detection_torch/object_detection_torch $APP_PATH

PLATFORM=x64-workstation
GPU=$(get_host_gpu)
if [ $(get_host_arch) == "aarch64" ]; then
    PLATFORM=igx-orin-devkit
fi

echo -e "done\n"
echo -e Use the following commands to package and run the Object Detection Torch application:
echo -e "Package the application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/object_detection_torch.yaml --platform [igx-orin-devkit | jetson-agx-orin-devkit | sbsa, x64-workstation] --platform-config [igpu | dgpu] -t holohub-object-detection-torch $APP_PATH/object_detection_torch --include onnx holoviz torch${NOCOLOR}"
echo -e "\nFor example:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/object_detection_torch.yaml --platform ${PLATFORM} --platform-config ${GPU} -t holohub-object-detection-torch $APP_PATH/object_detection_torch --include onnx holoviz torch${NOCOLOR}"
echo -e "\nRun the application:"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-object-detection-torch" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/object_detection_torch${NOCOLOR}"
