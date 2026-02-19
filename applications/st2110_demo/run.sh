#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 XRlabs. All rights reserved.
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

################################################################################
# ST2110 Demo Application Runner (Linux Sockets)
################################################################################
# This script runs the ST 2110-20 demo application in the NGC
# Holoscan container with all necessary environment variables configured.
#
# The demo shows all three output formats:
#   - raw_output: YCbCr-4:2:2-10bit (always enabled)
#   - rgba_output: RGBA 8-bit (configurable in YAML)
#   - nv12_output: NV12 8-bit (configurable in YAML)
#
# Prerequisites:
#   - ST2110 operator built: ./holohub build st2110_source
#   - Docker with NVIDIA runtime
#   - X11 display available
#   - ST 2110-20 stream arriving on configured multicast address
################################################################################

set -e

# Application and config files
DEMO_APP="st2110_demo_app.py"
CONFIG_FILE="st2110_demo_config.yaml"

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
HOLOHUB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${HOLOHUB_ROOT}/build/st2110_source"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}ST2110 Demo Application${NC}"
echo -e "${GREEN}================================${NC}"
echo

# Check if operator is built
if [[ ! -f "${BUILD_DIR}/operators/st2110_source/libst2110_source.so" ]]; then
    echo -e "${RED}ERROR: ST2110 operator not built!${NC}"
    echo
    echo "Please build the operator first:"
    echo "  cd ${HOLOHUB_ROOT}"
    echo "  ./holohub build st2110_source --configure-args '-DHOLOHUB_BUILD_PYTHON=ON'"
    exit 1
fi

# Check if config exists
if [[ ! -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]]; then
    echo -e "${RED}ERROR: Configuration file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting ST2110 Demo (Linux sockets)...${NC}"
echo "  App: ${DEMO_APP}"
echo "  Config: ${CONFIG_FILE}"
echo "  Container: nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13"
echo "  No root/DPDK required (uses standard Linux UDP sockets)"
echo ""
echo "Outputs (enable/disable in ${CONFIG_FILE}):"
echo "  - raw_output (YCbCr-4:2:2-10bit) - always enabled"
echo "  - rgba_output (RGBA 8-bit) - check enable_rgba_output"
echo "  - nv12_output (NV12 8-bit) - check enable_nv12_output"
echo

# Run the application in the container
docker run --rm --runtime=nvidia --network=host \
  --cap-add CAP_SYS_PTRACE --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${HOLOHUB_ROOT}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "/run/user/$(id -u):/run/user/$(id -u)" \
  -e "DISPLAY=$DISPLAY" \
  -e "XDG_RUNTIME_DIR=/run/user/$(id -u)" \
  -e PYTHONPATH=/opt/nvidia/holoscan/python/lib:/workspace/build/st2110_source/python/lib \
  -e LD_LIBRARY_PATH=/workspace/build/st2110_source/operators/st2110_source:/opt/nvidia/holoscan/lib \
  nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13 \
  bash -c "cd /workspace/applications/st2110_demo && python3 ${DEMO_APP} --config ${CONFIG_FILE}"

echo
echo -e "${GREEN}Application exited${NC}"
