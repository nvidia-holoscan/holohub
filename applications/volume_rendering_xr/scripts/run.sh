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

# Handle ML startup with conditional options
if [ -v ML_START_OPTIONS ]; then
    echo "Starting ML with options: ${ML_START_OPTIONS}"
    ml_start.sh ${ML_START_OPTIONS}
else
    echo "Starting ML with default debug options"
    ml_start.sh debug
fi

# Pair ML
echo "Pairing ML..."
ml_pair.sh

# Parse all arguments to check if xr_hello_holoscan is specified
USE_HELLO_HOLOSCAN=false
for arg in "$@"; do
    if [[ "$arg" == "xr_hello_holoscan" ]]; then
        USE_HELLO_HOLOSCAN=true
        break
    fi
done

if [ $USE_HELLO_HOLOSCAN = true ]; then
    # Run hello holoscan
    ./utils/xr_hello_holoscan/xr_hello_holoscan "$@"
else
    # Default: Run volume rendering
    ./volume_rendering_xr "$@"
fi
