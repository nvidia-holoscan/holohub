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
APP_PATH="$GIT_ROOT/install/imaging_ai_segmentator"

. $GIT_ROOT/utilities/bash_utils.sh

if [ ! -d $APP_PATH ]; then
    print_error "Please build the Imaging AI Segmentator application first with the following command:"
    print_error "./holohub install imaging_ai_segmentator"
    exit -1
fi

PLATFORM=$(get_platform_example_for_cli)

echo -e "\nPlease use the Holoscan SDK CLI to package and run the Imaging AI Segmentator application with the following set of commands:"
echo -e "\nGeneral command and options to package an application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/app.yaml --platform [jetson | igx-igpu | igx-dgpu | sbsa | x86_64] -t holohub-imaging_ai_segmentator -m $GIT_ROOT/data/imaging_ai_segmentator/models $APP_PATH/ ${NOCOLOR}"
echo -e "\nCommand to package this application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/app.yaml --platform ${PLATFORM}  -t holohub-imaging_ai_segmentator -m $GIT_ROOT/data/imaging_ai_segmentator/models $APP_PATH/ ${NOCOLOR}"
echo -e "\nList the newly built application container:"
echo -e "${YELLOW}docker images | grep "holohub-imaging_ai_segmentator" | awk '{print \$1\":\"\$2}' ${NOCOLOR}"
echo -e "\nRun the application container, after creating and cleaning output folder:"
echo -e "${YELLOW}mkdir -p ./output && rm -rf ./output/* ${NOCOLOR}"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-imaging_ai_segmentator" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/imaging_ai_segmentator/dicom -o ./output${NOCOLOR}"
echo -e "\n\nRefer to Packaging Holoscan Applications (https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information."
