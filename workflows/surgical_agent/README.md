# Surgical Agent Workflow

This directory contains a comprehensive workflow script that integrates the HoloHub video streaming application with the MONAI VLM-Surgical-Agent-Framework.

## Overview

The `surgical_agent.sh` script automates the following tasks:

1. **Video Streaming Server App**: Builds and runs the HoloHub WebRTC video streaming application
2. **Surgical Agent Framework**: Clones and runs the MONAI VLM-Surgical-Agent-Framework with Docker services

## Prerequisites

Before running the workflow, ensure you have:

- Docker and Docker Compose installed
- Git installed
- Sufficient GPU memory for the AI models (recommended: 10GB+ VRAM)
- Network access for downloading models and repositories

## Quick Start

### Run Everything (Default)

```bash
./surgical_agent.sh
```

or

```bash
./surgical_agent.sh all
```

This will:

1. Build the video streaming app
2. Start the video server at `http://127.0.0.1:8080`
3. Clone the VLM-Surgical-Agent-Framework repository
4. Start all surgical agent services via Docker

### Individual Components

#### Video Streaming Only

```bash
# Build the video app
./surgical_agent.sh build-video

# Run the video app
./surgical_agent.sh run-video
```

#### Surgical Agents Only

```bash
# Setup/clone the framework
./surgical_agent.sh setup-surgical

# Run the surgical agents
./surgical_agent.sh run-surgical
```

#### Management Commands

```bash
# Stop the video app
./surgical_agent.sh stop-video

# Clean up all services
./surgical_agent.sh clean

# Show help
./surgical_agent.sh help
```

## Services and Ports

When fully running, the following services will be available:

### Video Streaming

- **WebRTC Server**: `http://127.0.0.1:8080`
  - Web interface for video streaming
  - Click "Start" to begin video playback

### Surgical Agent Framework

The surgical agents framework runs multiple services via Docker:

- **LLM Server (vLLM)**: Port 8000
- **Whisper ASR**: Port 43001
- **Main Flask App**: Port 8050
- **TTS Service**: Various ports

Visit `http://127.0.0.1:8050` for the surgical agent interface.

## Repository Structure

After running, the directory will contain:

```text
workflows/surgical_agent/
├── surgical_agent.sh                    # Main workflow script
├── README.md                            # This file
├── metadata.json                        # The metadata file
└── VLM-Surgical-Agent-Framework/        # Cloned repository
    ├── agents/                          # Agent implementations
    ├── docker/                          # Docker configuration
    │   └── run-surgical-agents.sh       # Docker startup script
    ├── models/                          # AI model storage
    ├── servers/                         # Backend services
    └── web/                             # Frontend interface
```

## Integration Workflow

The typical workflow for using both systems together:

1. **Start Services**: Run `./surgical_agent.sh` to start both video streaming and surgical agents
2. **Video Analysis**:
   - Open `http://127.0.0.1:8080` for video streaming
   - Open `http://127.0.0.1:8050` for surgical analysis
3. **Interactive Analysis**: Use the surgical agent interface to analyze video content, take notes, and generate reports
4. **Cleanup**: Run `./surgical_agent.sh clean` when finished

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If services fail to start, check that ports 8080, 8000, 8050, and 43001 are available
2. **GPU Memory**: The surgical agents require significant GPU memory. Reduce model precision if needed
3. **Docker Issues**: Ensure Docker daemon is running and user has Docker permissions
4. **Network Issues**: VPN may interfere with WebRTC connections

### Logs and Debugging

- Video app logs: Check terminal output where `surgical_agent.sh` is running
- Docker logs: Use `docker logs <container_name>` for surgical agent services
- Service status: Use `docker ps` to check running containers

### Manual Cleanup

If the script cleanup fails:

```bash
# Stop video streaming containers specifically (using robust pattern matching)
docker stop $(docker ps --format "{{.ID}}\t{{.Image}}\t{{.Command}}" | awk '$2 ~ /^holohub:/ && $3 ~ /\.\/holohub run webrtc_video_server/ {print $1}')

# Stop all surgical agent containers
docker stop $(docker ps -q --filter "name=vlm-surgical")

# Stop any remaining holohub processes
pkill -f "holohub run webrtc_video_server"

# Stop all containers (if needed)
docker stop $(docker ps -q)
```

## References

- [HoloHub WebRTC Video Server](../../applications/webrtc_video_server/README.md)
- [VLM-Surgical-Agent-Framework](https://github.com/Project-MONAI/VLM-Surgical-Agent-Framework/)
