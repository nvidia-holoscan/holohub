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

# Create an array to store the background process IDs
declare -a bg_pids

# Base directory for the script
BASE_DIR="$(dirname "$(readlink -f "$0")")"

# Flags to control the script
LAUNCH_LOCAL=false
LAUNCH_MCP=false
ACTION_SET=false

# Function to clean up and kill the background processes
cleanup() {
    # Kill all background processes
    echo "Terminating background processes..."
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

# Function to build Holoscan vector DB
function build_db {
    mkdir -p "$BASE_DIR/embeddings"
    mkdir -p "$BASE_DIR/models"
    wget -nc -P "$BASE_DIR/docs/" https://developer.download.nvidia.com/assets/Clara/Holoscan_SDK_User_Guide_3.2.0.pdf
    if [ ! -f "$BASE_DIR/embeddings/holoscan/chroma.sqlite3" ]; then
        python3 build_holoscan_db.py
    fi
}

# Function to download the Phind LLM model from NGC
function download_llama {
    mkdir -p "$BASE_DIR/docs"
    if [ ! -f "$BASE_DIR/models/phind-codellama-34b-v2.Q5_K_M.gguf" ]; then
        wget -nc -P "$BASE_DIR/models" https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v2-GGUF/resolve/main/phind-codellama-34b-v2.Q5_K_M.gguf
    else
        echo "Model already exists, skipping download."
    fi
}

# Function that starts the appropriate server based on flags
function start_holochat {
    if [[ "$LAUNCH_LOCAL" == "true" ]]; then
        /workspace/llama.cpp/build/bin/server \
            -m "$BASE_DIR/models/phind-codellama-34b-v2.Q5_K_M.gguf" \
            --host 0.0.0.0 \
            -ngl 1000 \
            -c 4096 \
            -n 1024 \
        & bg_pids+=($!)
        python3 -u "$BASE_DIR/chatbot.py" --local \
        & bg_pids+=($!)
    elif [[ "$LAUNCH_MCP" == "true" ]]; then
        python3 -u "$BASE_DIR/chatbot.py" --mcp \
        & bg_pids+=($!)
    else
        python3 -u "$BASE_DIR/chatbot.py" \
        & bg_pids+=($!)
    fi
}

# Trap SIGINT (Ctrl-C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Parse any passed flags
for arg in "$@"; do
    case $arg in
        --local)
            LAUNCH_LOCAL=true
            ;;
        --mcp)
            LAUNCH_MCP=true
            ;;
        --build_llamaCpp | --build_db | --download_llama | --start_holochat)
            ACTION_SET=true
            ;;
    esac
done

# Process actions based on flags and user commands
if [ "$ACTION_SET" = false ]; then
    # If no specific function flags provided, run the full sequence
    build_db && \
    if [ "$LAUNCH_LOCAL" = true ]; then
        download_llama \
        && start_holochat
    elif [ "$LAUNCH_MCP" = true ]; then
        start_holochat
    else
        start_holochat
    fi
else
    for arg in "$@"; do
        case $arg in
            --build_llamaCpp)
                build_llamaCpp
                ;;
            --build_db)
                build_db
                ;;
            --download_llama)
                download_llama
                ;;
            --start_holochat)
                start_holochat
                ;;
        esac
    done
fi

# Let the script clean up the server process
wait
