#!/usr/bin/bash
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

# This script builds Holoscan SDK with debug symbols and runs the Endoscopy Tool Tracking application with GDB.
# Usage: ./debug_legacy.sh [build_type:debug,rel-debug,release] [holoscan_sdk_tag:v2.2.0] [holoscan_image:nvcr.io/nvidia/clara-holoscan/holoscan:v2.2.0-dgpu]

build_type=${1:-rel-debug}
holoscan_sdk_tag=${2:-v2.2.0}
holoscan_image=${3:-nvcr.io/nvidia/clara-holoscan/holoscan:v2.2.0-dgpu}

SCRIPT_DIR=$(dirname $(realpath $0))
HOLOHUB_ROOT=$(realpath "${SCRIPT_DIR}/../..")
tmp_dir=${SCRIPT_DIR}/tmp/holoscan_debugging
mkdir -p ${tmp_dir}

# Download and build Holoscan SDK
pushd ${tmp_dir}
if [ ! -d holoscan-sdk ]; then
    git clone git@github.com:nvidia-holoscan/holoscan-sdk.git
fi
cd holoscan-sdk
git checkout ${holoscan_sdk_tag}
./run build --type $build_type
INSTALL_DIR=$(realpath $(find . -type d -name "install-*"))
popd

# Build the tutorial container with GDB
${HOLOHUB_ROOT}/dev_container build \
    --img holohub:debugging \
    --docker_file ${SCRIPT_DIR}/Dockerfile \
    --base_img ${holoscan_image}

# Build the Endoscopy Tool Tracking application with debugging symbols
${HOLOHUB_ROOT}/dev_container launch \
    --docker_opts "-v ${INSTALL_DIR}:/opt/nvidia/holoscan" \
    --img holohub:debugging \
    -- bash -c './run build endoscopy_tool_tracking --type ${build_type}'

# Launch GDB with the Endoscopy Tool Tracking application
${HOLOHUB_ROOT}/dev_container launch \
    --docker_opts "-v ${INSTALL_DIR}:/opt/nvidia/holoscan --security-opt seccomp=unconfined" \
    --img holohub:debugging \
    --  bash -c \
            'gdb -q \
                -ex "break main" \
                -ex "run --data /workspace/holohub/data/endoscopy" \
                -ex "break /workspace/holoscan-sdk/src/core/application.cpp:add_flow" \
                /workspace/holohub/build/endoscopy_tool_tracking/applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking'
