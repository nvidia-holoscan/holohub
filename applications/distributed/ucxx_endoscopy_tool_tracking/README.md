# UCXX Endoscopy Tool Tracking - Distributed Application

A distributed implementation of the Endoscopy Tool Tracking application using UCXX (C++ interface to Unified Communication X) for high-performance network communication between nodes.

## Overview

This application demonstrates a distributed Holoscan pipeline that splits processing across multiple nodes:

- **Publisher**: Processes endoscopy video with LSTM-based tool tracking and broadcasts pre-rendered frames
- **Subscriber**: Receives and displays the pre-rendered frames with tracking overlays

The publisher performs the computationally expensive operations (video processing, AI inference, rendering) and transmits only the final rendered frames to lightweight subscriber(s) for display.

## Architecture

```
┌─────────────────────────────────────┐
│         PUBLISHER NODE              │
│                                     │
│  Video ──► Format ──► LSTM ──►     │
│          Converter    Inference     │
│                          │          │
│                          ▼          │
│                   Postprocessor     │
│                          │          │
│                          ▼          │
│                      Holoviz        │
│                    (Rendering)      │
│                          │          │
│                          ▼          │
│                  Format Converter   │
│                          │          │
│                          ▼          │
│                    UCXX Sender ─────┼──► Network
│                                     │
└─────────────────────────────────────┘

                    UCXX/UCX Protocol
                          │
                          ▼

┌─────────────────────────────────────┐
│        SUBSCRIBER NODE              │
│                                     │
│  Network ───► UCXX Receiver         │
│                    │                │
│                    ▼                │
│                 Holoviz             │
│               (Display)             │
│                    │                │
│                    ▼                │
│           Optional Recorder         │
│                                     │
└─────────────────────────────────────┘
```

## Key Features

- **High-Performance Communication**: Uses UCXX/UCX for low-latency, high-bandwidth data transfer
- **GPU-Direct RDMA Support**: Efficient GPU-to-GPU communication when available
- **Automatic Reconnection**: Handles network disconnections gracefully. Unlike standard Holoscan distributed fragments,
  subscriber applications can dynamically disconnect and reconnect without killing the publisher application.
- **Optional Recording**: Can record received frames for validation
- **Headless Operation**: Supports headless mode for testing

## Requirements

### Software
- Holoscan SDK 3.9.0 or later
- UCXX library (included with Holoscan)
- CUDA Toolkit
- CMake 3.20+

### Hardware
- NVIDIA GPU (tested on IGX)
- Network connectivity between publisher and subscriber nodes
- For optimal performance: RDMA-capable network hardware

### Data
- Endoscopy sample video dataset (automatically downloaded during build)

## Build and Run

### Publisher Node

Start the publisher on the machine with the video data and GPU for processing:

```bash
./run launch ucxx_endoscopy_tool_tracking publish
```

**Publisher Options:**
- `--data <path>` - Path to endoscopy video data (required)
- `--hostname <host>` - Hostname to bind (default: 0.0.0.0 - all interfaces)
- `--port <port>` - Port to listen on (default: 50008)
- `--config <path>` - Optional custom configuration file

### Subscriber Node

Start the subscriber on the display machine:

```bash
# Example connecting to localhost
./run launch ucxx_endoscopy_tool_tracking subscribe
```

**Subscriber Options:**
- `--mode subscribe` - Run as subscriber
- `--hostname <host>` - Publisher hostname/IP (default: 127.0.0.1)
- `--port <port>` - Publisher port (default: 50008)
- `--config <path>` - Optional custom configuration file

### Example: Same Machine Testing

```bash
# Terminal 1: Start publisher
./run launch ucxx_endoscopy_tool_tracking --mode publish

# Terminal 2: Start subscriber
./run launch ucxx_endoscopy_tool_tracking --mode subscribe
```

## Troubleshooting

### Subscriber can't connect to publisher

**Problem**: Subscriber logs "Attempting to reconnect" repeatedly

**Solutions**:
1. Verify publisher is running: `ps aux | grep ucxx_endoscopy`
2. Check network connectivity: `ping <publisher_ip>`
3. Verify port is not blocked: `telnet <publisher_ip> 50008`
4. Check firewall rules allow port 50008
5. Ensure both nodes use same port number

### No frames displayed

**Problem**: Subscriber connects but no visualization appears

**Solutions**:
1. Check logs - are frames being received?
2. Verify subscriber is not in headless mode
3. Check GPU availability on subscriber node
4. Ensure display environment is properly configured

## Related Applications

- **endoscopy_tool_tracking**: Single-node version
- **holoviz**: Visualization examples
- **ucxx_send_receive**: UCXX operator examples

## References

- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
- [UCXX Documentation](https://github.com/rapidsai/ucxx)
- [UCX Communication Framework](https://openucx.org/)
- [Endoscopy Tool Tracking Reference Application](/applications/endoscopy_tool_tracking/README.md)
