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

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# check if the service is running
if pgrep windrunner-serv >/dev/null; then
    source "${SCRIPT_DIR}/ml_stop.sh"
fi

if [ $# -eq 0 ]; then
    XRT_SKIP_STDIN='true' windrunner-service &
elif [ "$1" == "debug" ]; then
    XRT_SKIP_STDIN='true' XRT_DEBUG_GUI='true' XRT_LITE_UI='true'  windrunner-service &
fi

# check if the service is running
if ! ps -p $! > /dev/null; then
    echo "Failed to run windrunner service"
    exit 1
fi
# wait for service startup
until [ -e /tmp/windrunner_comp_ipc ] || [ -e /run/user/$(id -u)/windrunner_comp_ipc ];
do
     echo "Waiting for windrunner-service"
     sleep 1
done
