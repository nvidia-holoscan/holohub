# UCXX Endoscopy Tool Tracking - Distributed Application

## Overview

This application demonstrates distributed execution of the Endoscopy Tool Tracking pipeline using UCXX operators for high-performance inter-node communication. The application uses a **server-client architecture** where:

- **Server**: Replays video data, runs LSTM inference for tool tracking, displays results locally, and broadcasts processed frames to clients
- **Client**: Receives processed frames from the server and displays them

**Key Features:**
- Distributed processing with UCXX (Unified Communication X)
- High-performance, low-latency data transfer
- Server performs heavy computation (inference)
- Clients receive and display processed results
- Real-time visualization on both server and client

This application is built using Holoscan SDK version 3.8.0 and supports the following platforms: amd64, aarch64

## Architecture

### Server Pipeline
```
[Video Replayer] → [Format Converter] → [LSTM Inference] → [Tool Tracking Postprocessor]
                                                                         ↓
                                              ┌──────────────────────────┴────────────┐
                                              ↓                                        ↓
                                         [HolovizOp]                            [UCXX Sender]
                                     (Local Display)                          (Broadcast to Clients)
```

### Client Pipeline
```
[UCXX Receiver] → [HolovizOp]
 (from Server)     (Display)
```

## Prerequisites

- Holoscan SDK 3.8.0 or higher
- CUDA-capable GPU
- UCX library (for UCXX operators)
- Endoscopy tool tracking model data (for server only)
- Docker (for containerized deployment)

## Installation

### Build from Source

```bash
# From holohub root directory
./holohub build ucxx_endoscopy_tool_tracking
```

### Containerized Build

```bash
# Build the container
./holohub build-container ucxx_endoscopy_tool_tracking
```

## Usage

### Quick Start (Single Machine)

**Terminal 1 - Start Server:**
```bash
export HOLOSCAN_INPUT_PATH=/path/to/endoscopy/data
./ucxx_endoscopy_tool_tracking --mode server --data $HOLOSCAN_INPUT_PATH
```

**Terminal 2 - Start Client:**
```bash
./ucxx_endoscopy_tool_tracking --mode client --hostname localhost
```

### Multi-Node Deployment

**Server Node:**
```bash
# Run server on all network interfaces
export HOLOSCAN_INPUT_PATH=/path/to/endoscopy/data
./ucxx_endoscopy_tool_tracking --mode server --hostname 0.0.0.0 --port 50008 --data $HOLOSCAN_INPUT_PATH
```

**Client Node(s):**
```bash
# Connect to server (replace SERVER_IP with actual IP)
./ucxx_endoscopy_tool_tracking --mode client --hostname SERVER_IP --port 50008
```

### Command Line Options

```
Usage: ucxx_endoscopy_tool_tracking [options]

Options:
  -d, --data <path>        Path to data directory (required for server)
  -c, --config <path>      Path to config file (optional)
  -h, --hostname <host>    Hostname (default: 0.0.0.0 for server, 127.0.0.1 for client)
  -p, --port <port>        Port number (default: 50008)
  -m, --mode <mode>        Mode: 'server' or 'client' (required)
  -?, --help               Show this help message

Description:
  Server: Replays video, runs inference, displays locally, and broadcasts to clients
  Client: Receives processed frames from server and displays them

Examples:
  Server: ucxx_endoscopy_tool_tracking --mode server --data /path/to/data
  Client: ucxx_endoscopy_tool_tracking --mode client --hostname server_ip
```

### Environment Variables

- `HOLOSCAN_INPUT_PATH`: Path to endoscopy data directory (server only)
- `HOLOSCAN_CONFIG_PATH`: Path to configuration YAML file

## Configuration

The application uses `ucxx_endoscopy_tool_tracking.yaml` for configuration. Key sections:

### Server Configuration
- **replayer**: Video playback settings
- **format_converter**: Image preprocessing
- **lstm_inference**: TensorRT inference settings
- **holoviz**: Server-side visualization

### Client Configuration
- **holoviz_client**: Client-side visualization settings

Example configuration snippet:

```yaml
lstm_inference:
  force_engine_update: false
  verbose: true
  max_workspace_size: 2147483648
  enable_fp16_: true

holoviz:
  tensors:
    - name: ""
      type: color
    - name: mask
      type: color
    - name: scaled_coords
      type: crosses
```

## Network Configuration

### Firewall Settings

Ensure the server port is accessible:

```bash
# On server machine
sudo ufw allow 50008/tcp
```

### Connection Details

- **Protocol**: UCXX over TCP
- **Default Port**: 50008
- **Message Tag**: 1 (for processed frames)
- **Data Format**: Tool tracking postprocessor output (FlatBuffers)

## Data Requirements

### Server Requirements
- `tool_loc_convlstm.onnx`: LSTM model file
- `surgical_video.*`: Video data files for replayer
- Located in directory specified by `--data` or `HOLOSCAN_INPUT_PATH`

### Client Requirements
- No data files required (receives data from server)

## Project Structure

```
ucxx_endoscopy_tool_tracking/
├── CMakeLists.txt                          # Build configuration
├── Dockerfile                              # Container definition
├── README.md                               # This file
├── metadata.json                           # Application metadata
├── ucxx_endoscopy_tool_tracking.yaml      # Runtime configuration
└── src/
    ├── main.cpp                            # Entry point and argument parsing
    ├── server.cpp                          # Server application implementation
    ├── server.h                            # Server application header
    ├── client.cpp                          # Client application implementation
    └── client.h                            # Client application header
```

## Development

### Key Components

**Server:**
- `VideoStreamReplayerOp`: Replays endoscopy video
- `FormatConverterOp`: Preprocesses video frames
- `LSTMTensorRTInferenceOp`: Runs tool detection inference
- `ToolTrackingPostprocessorOp`: Post-processes inference results
- `HolovizOp`: Local visualization
- `UcxxSenderOp`: Broadcasts to clients

**Client:**
- `UcxxReceiverOp`: Receives processed frames
- `HolovizOp`: Displays results

### Extending the Application

**Add Processing on Client:**
Modify `client.cpp` to add operators between receiver and visualizer:

```cpp
auto custom_processor = make_operator<MyCustomOp>("processor");
add_flow(ucxx_receiver, custom_processor);
add_flow(custom_processor, holoviz);
```

**Broadcast Additional Data:**
Modify `server.cpp` to add additional UCXX senders with different tags.

## Troubleshooting

### Connection Issues

**Server won't start:**
- Check if port is already in use: `netstat -tulpn | grep 50008`
- Verify data directory is accessible
- Check CUDA/GPU availability

**Client can't connect:**
- Verify server is running and listening
- Check network connectivity: `ping server_ip`
- Verify port is not blocked by firewall
- Ensure hostname/IP is correct

### Performance Issues

**Low frame rate:**
- Monitor GPU utilization: `nvidia-smi`
- Check network bandwidth: `iperf3`
- Reduce visualization overhead if needed

**High latency:**
- Enable RDMA if available
- Adjust buffer sizes in configuration
- Check network quality

### Common Errors

**"Failed to load model"** (Server):
- Verify `HOLOSCAN_INPUT_PATH` points to correct directory
- Ensure `tool_loc_convlstm.onnx` exists

**"Endpoint closed with error"** (Client):
- Server may have stopped or crashed
- Check network stability
- Verify compatible UCXX versions

**"Schema mismatch"** (Client):
- Ensure server and client are using same schema version
- Rebuild both applications if needed

## Performance Metrics

Expected performance on NVIDIA RTX GPU:

**Server:**
- Inference latency: ~15-30ms per frame
- Frame rate: 20-30 FPS
- GPU memory: ~2-3GB

**Client:**
- Display latency: ~5-10ms
- Network latency: ~1-5ms (local), ~10-50ms (remote)
- Total end-to-end latency: ~50-100ms

## Use Cases

1. **Remote Visualization**: Run inference on powerful server, visualize on lightweight clients
2. **Multi-User Monitoring**: Multiple clients can observe same surgical procedure
3. **Load Distribution**: Offload heavy computation from edge devices
4. **Development/Debug**: Test inference pipeline on server while developing client UI

## License

This project is licensed under the Apache-2.0 License.

## Authors

- Holoscan Team - NVIDIA

## See Also

- [UCXX Operators Documentation](../../operators/ucxx_send_receive/README.md)
- [Endoscopy Tool Tracking Application](../../endoscopy_tool_tracking/cpp/README.md)
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Acknowledgments

- NVIDIA Holoscan Team
- UCX Development Team
- Open source community contributors
