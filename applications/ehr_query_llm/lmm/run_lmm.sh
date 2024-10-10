#!/bin/bash

# Array to hold PIDs of background processes
declare -a bg_pids

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

# Build ehr_query_llm/lmm
/workspace/holohub/run build ehr_query_llm/lmm

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
& python3 /workspace/holohub/applications/ehr_query_llm/lmm/asr_llm_tts.py \
& bg_pids+=($!) \

# Let the script clean up the server process
wait

