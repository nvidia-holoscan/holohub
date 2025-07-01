#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
APP_PATH="$GIT_ROOT/install/bin/object_detection_torch"

. $GIT_ROOT/utilities/bash_utils.sh

if [ ! -d $APP_PATH ]; then
    print_error "Please build the Object Detection Torch application first with the following command:"
    print_error "./holohub install object_detection_torch"
    exit -1
fi

PLATFORM=$(get_platform_example_for_cli)

echo -e Install Holoscan CLI and then use the following commands to package and run the Object Detection Torch application:
echo -e "Package the application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/object_detection_torch.yaml --platform [jetson | igx-igpu | igx-dgpu | sbsa | x86_64] -t holohub-object-detection-torch $APP_PATH/object_detection_torch --include onnx holoviz torch${NOCOLOR}"
echo -e "\nFor example:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/object_detection_torch.yaml --platform ${PLATFORM}  -t holohub-object-detection-torch $APP_PATH/object_detection_torch --include onnx holoviz torch${NOCOLOR}"
echo -e "\nRun the application:"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-object-detection-torch" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/object_detection_torch${NOCOLOR}"
echo -e "\n\nRefer to Packaging Holoscan Applications (https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information."
