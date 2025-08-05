#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
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

parse_arguments() {
    SKIP_MAGIC_LEAP=true
    USE_HELLO_HOLOSCAN=false
    COMMAND_ARGS=()
    
    for arg in "$@"; do
        case "$arg" in
            --magic-leap)
                SKIP_MAGIC_LEAP=false
                ;;
            xr_hello_holoscan)
                USE_HELLO_HOLOSCAN=true
                COMMAND_ARGS+=("$arg")
                ;;
            *)
                COMMAND_ARGS+=("$arg")
                ;;
        esac
    done
}

# Start Magic Leap if not skipped
start_magic_leap() {
    if [ -v ML_START_OPTIONS ]; then
        echo "Starting ML with options: ${ML_START_OPTIONS}"
        ml_start.sh ${ML_START_OPTIONS}
    else
        echo "Starting ML with default debug options"
        ml_start.sh debug
    fi
    
    echo "Pairing ML..."
    ml_pair.sh
}

run_application() {
    if [ "$USE_HELLO_HOLOSCAN" = true ]; then
        echo "Running XR Hello Holoscan..."
        ./utils/xr_hello_holoscan/xr_hello_holoscan "${COMMAND_ARGS[@]}"
    else
        echo "Running Volume Rendering XR..."
        ./volume_rendering_xr "${COMMAND_ARGS[@]}"
    fi
}

main() {
    parse_arguments "$@"
    
    if [ "$SKIP_MAGIC_LEAP" = false ]; then
        start_magic_leap
    fi
    
    run_application
}

main "$@"
