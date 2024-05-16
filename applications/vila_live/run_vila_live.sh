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

# Array to hold PIDs of background processes
declare -a bg_pids

# Function to clean up and kill the background processes
cleanup() {
    # Kill all background Python processes
    echo "Terminating background Python processes..."
    for pid in "${bg_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing process $pid..."
            kill $pid
        else
            echo "Process $pid does not exist."
        fi
    done

    exit 0
}

# Trap SIGINT (Ctrl-C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Run the Llama.cpp LLM server process + main HoloScrub app
python3 -m tinychat.serve.controller --host 0.0.0.0 --port 10000 & bg_pids+=($!)
python3 -m tinychat.serve.model_worker_new --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 \
    --model-path /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/ \
    --quant-path /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/llm/llama-3-vila1.5-8b-w4-g128-awq-v2.pt & bg_pids+=($!)
python3 /workspace/holohub/applications/vila_live/vila_live.py & bg_pids+=($!)

# Let the script clean up the server process
wait
