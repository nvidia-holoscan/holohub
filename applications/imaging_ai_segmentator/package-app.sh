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
APP_PATH="$GIT_ROOT/install/imaging_ai_segmentator"

. $GIT_ROOT/utilities/bash_utils.sh

if [ ! -d $APP_PATH ]; then
    print_error "Please build the Imaging AI Segmentator application first with the following command:"
    print_error "./dev_container build_and_run imaging_ai_segmentator --container_args \"-v $PWD/output:/var/holoscan/output\""
    exit -1
fi

PLATFORM=x64-workstation
GPU=$(get_host_gpu)
if [ $(get_host_arch) == "aarch64" ]; then
    PLATFORM=igx-orin-devkit
fi

echo -e "done\n"
echo -e Install Holoscan CLI and then use the following commands to package and run the Imaging AI Segmentator application:
echo -e "Package the application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/imaging_ai_segmentator.yaml --platform [igx-orin-devkit | jetson-agx-orin-devkit | sbsa, x64-workstation] --platform-config [igpu | dgpu] -t holohub-imaging_ai_segmentator -m $GIT_ROOT/data/imaging_ai_segmentator/models $APP_PATH/"
echo -e "\nFor example:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/imaging_ai_segmentator.yaml --platform ${PLATFORM} --platform-config ${GPU} -t holohub-imaging_ai_segmentator -m $GIT_ROOT/data/imaging_ai_segmentator/models $APP_PATH/"
echo -e "\nRun the application:"
echo -e "mkdir ./output"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-imaging_ai_segmentator" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/imaging_ai_segmentator/dicom -o ./output${NOCOLOR}"
echo -e "\n\nRefer to Packaging Holoscan Applications (https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information."
