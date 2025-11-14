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
# This script builds and runs the WebRTC video streaming app with camera support,
# clones the VLM-Surgical-Agent-Framework, and starts the surgical agents services

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

# Container name for WebRTC video streaming server
WEBRTC_CONTAINER_NAME="holohub_surgical_video"

# Helper function to find WebRTC video streaming containers
find_video_streaming_containers() {
    # Find containers by name - much more robust than inspecting commands
    docker ps -q --filter "name=^/${WEBRTC_CONTAINER_NAME}$" || true
}

# Helper function to check if video streaming containers are running
is_video_streaming_running() {
    [ -n "$(find_video_streaming_containers)" ]
}

# Function to build video streaming app
build_video_app() {
    print_status "Building WebRTC video streaming server app..."
    cd "$HOLOHUB_ROOT"

    if ! ./holohub build webrtc_video_server; then
        print_error "Failed to build WebRTC video streaming server app"
        return 1
    fi

    print_success "WebRTC video streaming server built successfully"
}

# Function to run video streaming app
run_video_app() {
    # Stop any existing instance first
    stop_video_app

    print_status "Starting WebRTC video streaming server with camera mode..."
    cd "$HOLOHUB_ROOT"

    # Camera device configuration (can be overridden via environment variable)
    CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
    PIXEL_FORMAT="${PIXEL_FORMAT:-YUYV}"

    print_status "Using camera device: $CAMERA_DEVICE"
    print_status "Using pixel format: $PIXEL_FORMAT"
    print_status "Container name: $WEBRTC_CONTAINER_NAME"

    # Run in background with camera mode
    # Note: --source camera enables V4L2 camera capture
    # --no-serve-client runs in API-only mode (no embedded web UI)
    # --docker-opts sets the container name for easy identification
    ./holohub run webrtc_video_server \
        --docker-opts="--name $WEBRTC_CONTAINER_NAME" \
        --run-args="--source camera --camera-device $CAMERA_DEVICE --pixel-format $PIXEL_FORMAT --no-serve-client" &

    print_success "WebRTC video streaming app started in camera mode"
    print_status "Video server API available at http://127.0.0.1:8080"
    print_status "API endpoints: POST /offer, GET /iceServers"

    # Wait a moment for the container to start
    sleep 5

    # Verify container is running
    if is_video_streaming_running; then
        print_success "WebRTC video streaming container is running"
    else
        print_warning "WebRTC video streaming container may not have started properly"
    fi
}

# Function to clone surgical agent framework
clone_surgical_framework() {
    print_status "Cloning VLM-Surgical-Agent-Framework..."
    cd "$WORKFLOW_DIR"

    if [ -d "VLM-Surgical-Agent-Framework" ]; then
        print_warning "VLM-Surgical-Agent-Framework directory already exists. Pulling latest changes..."
        cd VLM-Surgical-Agent-Framework
        if ! git pull; then
            print_warning "Failed to pull latest changes. Repository may have local modifications."
            print_status "Continuing with existing repository state..."
        fi
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
    print_status "Stopping WebRTC video streaming containers..."

    # Find containers by name (both running and stopped)
    local containers
    containers=$(docker ps -aq --filter "name=^/${WEBRTC_CONTAINER_NAME}$")

    if [ -n "$containers" ]; then
        echo "$containers" | while read -r container; do
            if [ -n "$container" ]; then
                local container_name
                container_name=$(docker ps -a --format "{{.Names}}" --filter "id=$container")
                local container_status
                container_status=$(docker ps -a --format "{{.Status}}" --filter "id=$container")
                print_status "Found container: $container_name ($container) - Status: $container_status"

                # Stop if running
                if docker ps -q --filter "id=$container" | grep -q .; then
                    print_status "Stopping container: $container_name"
                    if docker stop "$container" 2>/dev/null; then
                        print_success "Stopped container: $container_name"
                    else
                        print_warning "Failed to stop container: $container_name"
                    fi
                fi

                # Remove the container to free up the name
                # Use -f to force removal in case container is still shutting down
                print_status "Removing container: $container_name"
                if docker rm -f "$container" >/dev/null 2>&1; then
                    print_success "Removed container: $container_name"
                else
                    print_warning "Failed to remove container: $container_name"
                fi
            fi
        done
    else
        print_status "No WebRTC video streaming containers found"
    fi

    # Also terminate any holohub CLI processes as backup cleanup
    if pkill -f "holohub run webrtc_video_server" 2>/dev/null; then
        print_status "Terminated holohub CLI processes"
    fi

    # Clean up legacy PID file if it exists
    if [ -f "$WORKFLOW_DIR/video_app.pid" ]; then
        rm -f "$WORKFLOW_DIR/video_app.pid"
        print_status "Removed legacy PID file"
    fi

    print_success "WebRTC video streaming app cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all                 Build and run everything (default)"
    echo "  build-video         Build WebRTC video streaming server app only"
    echo "  run-video           Run WebRTC video streaming server app with camera only"
    echo "  setup-surgical      Clone/update surgical framework only"
    echo "  run-surgical        Run surgical agents only"
    echo "  stop-video          Stop WebRTC video streaming server app"
    echo "  stop-surgical       Stop surgical agents services"
    echo "  stop-all            Stop all services"
    echo "  clean               Stop all services and remove all artifacts"
    echo "  help                Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CAMERA_DEVICE       Camera device path (default: /dev/video0)"
    echo "  PIXEL_FORMAT        Camera pixel format (default: YUYV)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run everything"
    echo "  $0 all                                # Run everything"
    echo "  $0 build-video                        # Just build the WebRTC video server app"
    echo "  $0 stop-video                         # Stop the WebRTC video server app"
    echo "  $0 stop-all                           # Stop all services"
    echo "  $0 clean                              # Stop all services and remove cloned repos"
    echo "  CAMERA_DEVICE=/dev/video5 $0 run-video  # Use different camera device"
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

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    stop_video_app
    stop_surgical_agents
    print_success "All services stopped"
}

# Function to clean up everything
cleanup() {
    print_status "Cleaning up all services and artifacts..."
    stop_video_app
    stop_surgical_agents

    # Remove cloned surgical framework
    if [ -d "$WORKFLOW_DIR/VLM-Surgical-Agent-Framework" ]; then
        print_status "Removing VLM-Surgical-Agent-Framework directory..."
        rm -rf "$WORKFLOW_DIR/VLM-Surgical-Agent-Framework"
        print_success "Removed VLM-Surgical-Agent-Framework directory"
    fi

    print_success "Complete cleanup finished"
}

# Trap to stop all services on script interruption/error (but not normal exit)
trap stop_all INT TERM

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
        print_status "WebRTC video app running in camera mode. Press Ctrl+C to stop."

        # Wait for containers to be running, then wait for user interruption
        while is_video_streaming_running; do
            sleep 5
        done
        print_warning "WebRTC video streaming container stopped unexpectedly"
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
    "stop-all")
        stop_all
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
print_status "- WebRTC video streaming server (camera mode): http://127.0.0.1:8080"
print_status "  API Endpoints: POST /offer, GET /iceServers"
print_status "- Surgical agents: Check docker container status with 'docker ps'"
print_status ""
print_status "To stop the video app: $0 stop-video"
print_status "To stop surgical agents: $0 stop-surgical"
print_status "To stop all services: $0 stop-all"
print_status "To clean up everything (stop + remove artifacts): $0 clean"
