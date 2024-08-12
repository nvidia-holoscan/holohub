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

GIT_ROOT=$(readlink -f ./$(git rev-parse --show-cdup))

# Utilities

YELLOW="\e[1;33m"
RED="\e[1;31m"
NOCOLOR="\e[0m"

print_error() {
    echo -e "${RED}ERROR${NOCOLOR}:" $*
}

if [ ! -d $GIT_ROOT/build/endoscopy_tool_tracking ]; then
    print_error "Please build the Endoscopy Tool Tracking application first with the following command:"
    print_error "./dev_container build_and_run endoscopy_tool_tracking"
    exit -1
fi

APP_PATH="$GIT_ROOT/endoscopy_tool_tracking_python"
echo Creating application directory $APP_PATH...
mkdir -p $APP_PATH
echo Copying application files to $APP_PATH...
cp -f $GIT_ROOT/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py $APP_PATH
cp -f $GIT_ROOT/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.yaml $APP_PATH
cp -f $GIT_ROOT/build/endoscopy_tool_tracking/gxf_extensions/lstm_tensor_rt_inference/*.so $APP_PATH
cp -f $GIT_ROOT/build/endoscopy_tool_tracking/operators/lstm_tensor_rt_inference/liblstm_tensor_rt_inference.so $APP_PATH
cp -f $GIT_ROOT/build/endoscopy_tool_tracking/operators/tool_tracking_postprocessor/libtool_tracking_postprocessor.so $APP_PATH
cp -rf $GIT_ROOT/build/endoscopy_tool_tracking/python/lib/holohub $APP_PATH
echo Updating application configuration...
sed -e s!gxf_extensions/lstm_tensor_rt_inference/!!g -i $APP_PATH/endoscopy_tool_tracking.yaml


echo -e "done\n"
echo -e Use the following commands to package and run the Endoscopy Tool Tracking application:\n
echo -e "Package the application:\n"
echo -e "${YELLOW}holoscan package -c $APP_PATH/endoscopy_tool_tracking.yaml --platform x64-workstation -t holohub-endoscopy-tool-tracking-python $APP_PATH/endoscopy_tool_tracking.py --include onnx holoviz${NOCOLOR}"
echo -e "Run the application:\n"
echo -e "${YELLOW}holoscan run -r \$(docker images | grep "holohub-endoscopy-tool-tracking-python" | awk '{print \$1\":\"\$2}') -i $GIT_ROOT/data/endoscopy${NOCOLOR}"
