#!/bin/bash

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

set_python_path() {
    # Set the python path to the holoscan_sdk_install path
    local holoscan_sdk_install=$(grep -Po '^holoscan_DIR:PATH=\K[^ ]*' /workspace/holohub/build/CMakeCache.txt)
    local holohub_build_dir="${SCRIPT_DIR}/build"
    local environment="export PYTHONPATH=${PYTHONPATH}:${holoscan_sdk_install}/../../../python/lib:${holohub_build_dir}/python/lib:${SCRIPT_DIR}"
    echo "${environment}"
    eval "${environment}"
}

# Trap SIGINT (Ctrl-C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Set environment variables
set_python_path

# Run the Llama.cpp LLM server process + main HoloScrub app
python3 -m tinychat.serve.controller --host 0.0.0.0 --port 10000 & bg_pids+=($!)
python3 -m tinychat.serve.model_worker_new --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 \
    --model-path /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/ \
    --quant-path /workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/llm/llama-3-vila1.5-8b-w4-g128-awq-v2.pt & bg_pids+=($!)
python3 /workspace/holohub/applications/vila_live/vila_live.py & bg_pids+=($!)

# Let the script clean up the server process
wait
