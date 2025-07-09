#!/bin/bash
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
set -e

GIT_ROOT=$(readlink -f ./$(git rev-parse --show-cdup))
APP_PATH="$GIT_ROOT/install/bin/grpc_endoscopy_tool_tracking/python"

. $GIT_ROOT/utilities/bash_utils.sh

if [ ! -d $APP_PATH ]; then
    print_error "Please build the gRPC Endoscopy Tool Tracking application first with the following command:"
    print_error "./holohub install grpc_endoscopy_tool_tracking"
    exit -1
fi

PLATFORM=x64-workstation
GPU=$(get_host_gpu)
if [ $(get_host_arch) == "aarch64" ]; then
    PLATFORM=igx-orin-devkit
fi

echo -e "Modifying configuration file for packaging..."
sed -i 's|lib/gxf_extensions/||' "$APP_PATH/endoscopy_tool_tracking.yaml"
echo -e "done\n"

echo -e Install Holoscan CLI and then use the following commands to package and run the Endoscopy Tool Tracking application:
echo -e "==========Package the application=========="
echo -e "Cloud:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/endoscopy_tool_tracking.yaml --platform [igx-orin-devkit | jetson-agx-orin-devkit | sbsa, x64-workstation] --platform-config [igpu | dgpu] -t holohub-grpc-endoscopy-tool-tracking-cloud $APP_PATH/cloud/app_cloud_main.py --include onnx holoviz$ --add $GIT_ROOT/install/lib --add $GIT_ROOT/install/python${NOCOLOR}"
echo -e "\nFor example:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/endoscopy_tool_tracking.yaml --platform ${PLATFORM} --platform-config ${GPU} -t holohub-grpc-endoscopy-tool-tracking-cloud $APP_PATH/cloud/app_cloud_main.py --include onnx holoviz --add $GIT_ROOT/install/lib --add $GIT_ROOT/install/python${NOCOLOR}"
echo -e "\nEdge:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/endoscopy_tool_tracking.yaml --platform [igx-orin-devkit | jetson-agx-orin-devkit | sbsa, x64-workstation] --platform-config [igpu | dgpu] -t holohub-grpc-endoscopy-tool-tracking-edge $APP_PATH/edge/app_edge_main.py --include onnx holoviz --add $GIT_ROOT/install/lib --add $GIT_ROOT/install/python${NOCOLOR}"
echo -e "\nFor example:"
echo -e "${YELLOW}holoscan package -c $APP_PATH/endoscopy_tool_tracking.yaml --platform ${PLATFORM} --platform-config ${GPU} -t holohub-grpc-endoscopy-tool-tracking-edge $APP_PATH/edge/app_edge_main.py --include onnx holoviz --add $GIT_ROOT/install/lib --add $GIT_ROOT/install/python${NOCOLOR}"
echo -e "\n\n==========Run the application=========="
echo -e "Cloud:"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-grpc-endoscopy-tool-tracking-cloud" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/endoscopy${NOCOLOR}"
echo -e "\nEdge:"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-grpc-endoscopy-tool-tracking-edge" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/endoscopy${NOCOLOR}"
echo -e "\n\nRefer to Packaging Holoscan Applications (https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information."
