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

app=""
args=()

run_command() {
    local status=0
    local cmd="$*"

    echo -e "${YELLOW}[command]${NOCOLOR} ${cmd}"

    [ "$(echo -n "$@")" = "" ] && return 1 # return 1 if there is no command available

    if [ "${DO_DRY_RUN}" != "true" ]; then
        eval "$@"
        status=$?
    fi
    return $status
}


while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--"* ]]; then
        args+=("$1" "$2")
        shift 2
    elif [ -z "$app" ]; then
        app=$1
        shift 1
    else
        echo "Invalid argument: $1"
        exit 1
    fi
done

if [ "$app" == "cloud" ]; then
    run_command ./grpc_endoscopy_tool_tracking_cloud "${args[@]}"
elif [ "$app" == "edge" ]; then
    run_command ./grpc_endoscopy_tool_tracking_edge "${args[@]}"
else
    echo "Invalid application: ${app:-<not specified>}"
    exit 1
fi
