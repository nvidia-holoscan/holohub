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

# This script runs the Endsocopy Tool Tracking application for interactive debugging with GDB.
# Usage: ./debug_gdb.sh [build_type:debug,rel-debug,release]

build_type=${1:-debug}

SCRIPT_DIR=$(dirname $(realpath $0))
HOLOHUB_ROOT=$(realpath "${SCRIPT_DIR}/../..")
tmp_dir=$(pwd)/tmp/holoscan_debugging
mkdir -p ${tmp_dir}

# Build the tutorial container with GDB
${HOLOHUB_ROOT}/dev_container build \
    --img holohub:debugging \
    --docker_file ${SCRIPT_DIR}/Dockerfile \
    --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.3.0-dgpu

# Build the Endoscopy Tool Tracking application with debugging symbols
${HOLOHUB_ROOT}/dev_container launch \
    --img holohub:debugging \
    -- ./run build endoscopy_tool_tracking --type ${build_type}

# Launch GDB with the Endoscopy Tool Tracking application
${HOLOHUB_ROOT}/dev_container launch \
    --img holohub:debugging \
    --docker_opts "--security-opt seccomp=unconfined" \
    --  bash -c '\
            export PYTHONPATH=/opt/nvidia/holoscan/python/lib:/workspace/holohub/benchmarks/holoscan_flow_benchmarking:/usr/share/gdb/python && \
            gdb \
                -ex "break main" \
                -ex "run --data /workspace/holohub/data/endoscopy" \
                -ex "break /workspace/holoscan-sdk/src/core/application.cpp:add_flow" \
                /workspace/holohub/build/endoscopy_tool_tracking/applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking'
