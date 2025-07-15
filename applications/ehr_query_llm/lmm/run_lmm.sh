#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
# reserved. SPDX-License-Identifier: Apache-2.0
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

set -x

# Function to clean up and kill the background processes
cleanup() {
    echo "Terminating Llama.cpp server process..."
    # Kill the /bin/server process.
    pkill -f "/opt/nvidia/holoscan/llama.cpp/build/bin/server"

    # Kill all background Python processes
    for pid in "${bg_pids[@]}"; do
        if kill -0 $pid > /dev/null 2>&1; then
            kill $pid
        fi
    done

    exit 0
}

set_python_path() {
    # Set the python path to the holoscan_sdk_install path
    local holoscan_sdk_install=$(grep -Po '^holoscan_DIR:PATH=\K[^ ]*' build/CMakeCache.txt)
    local holohub_build_dir="${SCRIPT_DIR}/build"
    local environment="export PYTHONPATH=${holoscan_sdk_install}/../../../python/lib:${holohub_build_dir}/python/lib:${SCRIPT_DIR}"
    # Check that Holoscan is available in PYTHONPATH
    python -c "import holoscan" || ("Failed to import Holoscan module" && exit 1)
    echo "${environment}"
    eval "${environment}"
}

set_transformer_cache() {
    local transformer_cache="export HF_HOME=/workspace/volumes/models"
    echo "${transformer_cache}"
    eval "${transformer_cache}"
}

set_offline_flags() {
    local offline_flags="export TRANSFORMERS_OFFLINE=1 ANONYMIZED_TELEMETRY=False"
    echo "${offline_flags}"
    eval "${offline_flags}"
}

# Trap SIGINT (Ctrl-C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Set environment variables
set_python_path
set_transformer_cache
# set_offline_flags
netstat -tuln | grep -E ':8080|:8050|:49000'

# Build ehr_query_llm/lmm
/workspace/holohub/holohub build ehr_query_llm/lmm

# Define log file paths
LLM_LOG_FILE="/workspace/holohub/applications/ehr_query_llm/lmm/llama_cpp.log"

echo "Hold on tight, starting the Llama.cpp LLM server process + main EHR app!"

# Run the Llama.cpp LLM server process + main EHR app
/opt/nvidia/holoscan/llama.cpp/build/bin/server \
    -m  /workspace/volumes/models/openchat-3.5-0106.Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 1000 \
    -c 8000 \
    --cont-batching \
    > ${LLM_LOG_FILE} 2>&1 \
& python3 /workspace/holohub/applications/ehr_query_llm/lmm/asr_llm_tts.py &
bg_pids+=($!)

sleep 5  # Give the server time to start
if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q "200"; then
    echo "Llama.cpp server failed to start or respond"
    exit 1
fi

echo "Background processes:"
jobs -l

# Add a delay
#sleep 2

# Let the script clean up the server process
wait

