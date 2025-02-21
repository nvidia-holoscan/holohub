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

# Get an example value for the running PLATFORM.
# Since we cannot detect the actual hardware, IGX vs Jetson, we use IGX as an example.
get_platform_example_for_cli() {
    if [ $(get_host_arch) == "aarch64" ]; then
        if [ $(get_host_gpu) == "igpu" ]; then
            PLATFORM=igx-igpu
        else
            PLATFORM=igx-dgpu
        fi
    else
        PLATFORM=x86_64
    fi
    echo -n $PLATFORM
}