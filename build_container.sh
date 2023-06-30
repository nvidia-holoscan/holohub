#!/bin/bash 
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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



if [ "$0" != "/bin/bash" ] && [ "$0" != "bash" ]; then
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    export RUN_SCRIPT_FILE="$(readlink -f "$0")"
else
    export RUN_SCRIPT_FILE="$(readlink -f "${BASH_SOURCE[0]}")"
fi

export TOP=$(dirname "${RUN_SCRIPT_FILE}")
export HOLOHUB_ROOT=${TOP}


export DOCKER_BUILDKIT=1

docker build  \
        --build-arg BUILDKIT_INLINE_CACHE=0 \
        --network=host \
        -t holohub:sdk_0.5.1  ${TOP}
