#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Surgical Agent Workflow Script
# This script builds and runs the video streaming app, clones the VLM-Surgical-Agent-Framework,
# and starts the surgical agents services

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the script directory and holohub root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
HOLOHUB_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
WORKFLOW_DIR="$SCRIPT_DIR"

print_status "Script directory: $SCRIPT_DIR"
print_status "Holohub root: $HOLOHUB_ROOT"

# Helper function to find live video streaming containers
find_video_streaming_containers() {
    # Find containers with the live_video_streaming_server image tag
    # The image tag uniquely identifies these containers
    docker ps --filter "ancestor=holohub:live_video_streaming_server" --format "{{.ID}}" || true
}

# Helper function to check if video streaming containers are running
is_video_streaming_running() {
    [ -n "$(find_video_streaming_containers)" ]
}

# Function to build video streaming app
build_video_app() {
    print_status "Building video streaming server app..."
    cd "$HOLOHUB_ROOT"

    if ! ./holohub build live_video_streaming_server; then
        print_error "Failed to build video streaming server app"
        return 1
    fi

    print_success "Video streaming server built successfully"
}

# Function to run video streaming app
run_video_app() {
    # Stop any existing instance first
    stop_video_app

    print_status "Starting video streaming server app..."
    cd "$HOLOHUB_ROOT"

    # Run in background
    ./holohub run live_video_streaming_server &

    print_success "Video streaming app started"
    print_status "Video server should be available at http://127.0.0.1:8080"

    # Wait a moment for the container to start
    sleep 5

    # Verify container is running
    if is_video_streaming_running; then
        print_success "Video streaming container is running"
    else
        print_warning "Video streaming container may not have started properly"
    fi
}

# Function to clone surgical agent framework
clone_surgical_framework() {
    print_status "Cloning VLM-Surgical-Agent-Framework..."
    cd "$WORKFLOW_DIR"

    if [ -d "VLM-Surgical-Agent-Framework" ]; then
        print_warning "VLM-Surgical-Agent-Framework directory already exists. Pulling latest changes..."
        cd VLM-Surgical-Agent-Framework
        git pull
    else
        git clone https://github.com/Project-MONAI/VLM-Surgical-Agent-Framework.git
        cd VLM-Surgical-Agent-Framework
    fi

    print_success "VLM-Surgical-Agent-Framework repository ready"
}

# Function to run surgical agents
run_surgical_agents() {
    print_status "Starting surgical agents services..."
    cd "$WORKFLOW_DIR/VLM-Surgical-Agent-Framework/docker"

    if [ ! -f "run-surgical-agents.sh" ]; then
        print_error "run-surgical-agents.sh not found in docker directory"
        return 1
    fi

    # Make the script executable
    chmod +x run-surgical-agents.sh

    print_status "Running surgical agents docker services..."
    ./run-surgical-agents.sh
}

# Function to stop video app
stop_video_app() {
    print_status "Stopping video streaming containers..."

    # Find containers running holohub with live_video_streaming_server
    local containers=$(find_video_streaming_containers)

    if [ -n "$containers" ]; then
        echo "$containers" | while read -r container; do
            if [ -n "$container" ]; then
                local container_name=$(docker ps --format "{{.Names}}" --filter "id=$container")
                print_status "Stopping container: $container_name ($container)"
                if docker stop "$container" 2>/dev/null; then
                    print_success "Stopped container: $container_name"
                else
                    print_warning "Failed to stop container: $container_name"
                fi
            fi
        done
    else
        print_status "No video streaming containers found running"
    fi

    # Also terminate any holohub CLI processes as backup cleanup
    if pkill -f "holohub run live_video_streaming_server" 2>/dev/null; then
        print_status "Terminated holohub CLI processes"
    fi

    # Clean up legacy PID file if it exists
    if [ -f "$WORKFLOW_DIR/video_app.pid" ]; then
        rm -f "$WORKFLOW_DIR/video_app.pid"
        print_status "Removed legacy PID file"
    fi

    print_success "Video streaming app cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all                 Build and run everything (default)"
    echo "  build-video         Build video streaming server app only"
    echo "  run-video           Run video streaming server app only"
    echo "  setup-surgical      Clone/update surgical framework only"
    echo "  run-surgical        Run surgical agents only"
    echo "  stop-video          Stop video streaming server app"
    echo "  stop-surgical       Stop surgical agents services"
    echo "  clean               Stop all services and clean up"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run everything"
    echo "  $0 all              # Run everything"
    echo "  $0 build-video      # Just build the video streaming server app"
    echo "  $0 stop-video       # Stop the video streaming server app"
}

# Function to stop surgical agents
stop_surgical_agents() {
    print_status "Stopping surgical agents services..."
    cd "$WORKFLOW_DIR/VLM-Surgical-Agent-Framework/docker"

    if [ -f "run-surgical-agents.sh" ]; then
        print_status "Running surgical agents stop command..."
        ./run-surgical-agents.sh stop
        print_success "Surgical agents services stopped"
    else
        print_warning "run-surgical-agents.sh not found, cannot stop surgical agents"
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up all services..."
    stop_video_app
    stop_surgical_agents
    print_success "All services cleanup completed"
}

# Trap to cleanup on script interruption/error (but not normal exit)
trap cleanup INT TERM

# Main execution
case "${1:-all}" in
    "all")
        print_status "Starting complete surgical agent workflow..."
        build_video_app
        run_video_app
        clone_surgical_framework
        run_surgical_agents
        ;;
    "build-video")
        build_video_app
        ;;
    "run-video")
        run_video_app
        # Keep script running to maintain the background process
        print_status "Video app running. Press Ctrl+C to stop."

        # Wait for containers to be running, then wait for user interruption
        while is_video_streaming_running; do
            sleep 5
        done
        print_warning "Video streaming container stopped unexpectedly"
        ;;
    "setup-surgical")
        clone_surgical_framework
        ;;
    "run-surgical")
        run_surgical_agents
        ;;
    "stop-video")
        stop_video_app
        exit 0
        ;;
    "stop-surgical")
        stop_surgical_agents
        exit 0
        ;;
    "clean")
        cleanup
        exit 0
        ;;
    "help")
        show_usage
        exit 0
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

print_success "Workflow completed successfully!"
print_status "Services running:"
print_status "- Video streaming server: http://127.0.0.1:8080"
print_status "- Surgical agents: Check docker container status with 'docker ps'"
print_status ""
print_status "To stop the video app: $0 stop-video"
print_status "To stop surgical agents: $0 stop-surgical"
print_status "To clean up all services: $0 clean"
