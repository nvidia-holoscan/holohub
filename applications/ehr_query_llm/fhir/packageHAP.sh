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
APP_PATH="$GIT_ROOT/install/bin/fhir/python"
APP_CONFIG="app.yaml"
APP_NAME="fhir"
IMAGE_TAG="holohub-fhir"

. $GIT_ROOT/utilities/bash_utils.sh

if [ ! -d $APP_PATH ]; then
    print_error "Please build the application first with the following command:"
    print_error "./dev_container build_and_install $APP_NAME"
    exit -1
fi

PLATFORM=x64-workstation
GPU=$(get_host_gpu)
if [ $(get_host_arch) == "aarch64" ]; then
    PLATFORM=igx-orin-devkit
fi

echo -e "\nPlease use the Holoscan SDK CLI to package and run the $APP_NAME application container with the following set of commands:"
echo -e "\nGeneral command and options to package an application:"
echo -e "${YELLOW}holoscan package -c <App_Path>/<App_Config> --platform [igx-orin-devkit | jetson-agx-orin-devkit | sbsa, x64-workstation] --platform-config [igpu | dgpu] -t <Image Tag> ${NOCOLOR}"
echo -e "\nCommand to package this application:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/$APP_CONFIG --platform ${PLATFORM} --platform-config ${GPU} -t $IMAGE_TAG $APP_PATH ${NOCOLOR}"
echo -e "\nList the newly built application container image (command result also shown):"
echo -e "${YELLOW}docker images | grep "$IMAGE_TAG" | awk '{print \$1\":\"\$2}' ${NOCOLOR}"
IMAGE_FULL_NAME=$(docker images | grep "$IMAGE_TAG" | awk '{print $1":"$2}')
echo -e "$IMAGE_FULL_NAME"
echo -e "\nRun the application container with your own FHIR server URL, and if required, the OAUth2 URL and client info:"
echo -e "${YELLOW}docker run -it --rm --net host $IMAGE_FULL_NAME --fhir_url <url> [--auth_url <url> --uid <id> --secret <token>] ${NOCOLOR}"
echo -e "\n\nRefer to Packaging Holoscan Applications (https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information."
