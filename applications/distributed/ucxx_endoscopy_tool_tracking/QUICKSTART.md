# UCXX Endoscopy Tool Tracking - Quick Start Guide

## Overview

This distributed application uses a **server-client architecture**:
- **Server**: Processes video and broadcasts results
- **Client**: Receives and displays processed results

Perfect for remote visualization, multi-user monitoring, and distributed inference.

## Quick Start - Single Machine

### Step 1: Build

```bash
cd /path/to/holohub
./holohub build ucxx_endoscopy_tool_tracking
cd build/applications/distributed/ucxx_endoscopy_tool_tracking
```

### Step 2: Start Server

```bash
# Terminal 1
export HOLOSCAN_INPUT_PATH=/path/to/endoscopy/data
./ucxx_endoscopy_tool_tracking --mode server --data $HOLOSCAN_INPUT_PATH
```

Expected output:
```
[INFO] Starting SERVER: Processing video and broadcasting to clients
[INFO] Composing SERVER application - processing and broadcasting frames
[INFO] Server pipeline: Replayer → Processing → Local Display + Broadcast
```

### Step 3: Start Client

```bash
# Terminal 2
./ucxx_endoscopy_tool_tracking --mode client --hostname localhost
```

Expected output:
```
[INFO] Starting CLIENT: Receiving processed frames from server
[INFO] Composing CLIENT application - receiving and displaying processed frames
[INFO] Client pipeline: Receive → Display
```

You should now see:
- **Server window**: Video with tool tracking overlays (rendered locally)
- **Client window**: Same rendered frame as server (broadcasted over network)

## Architecture Overview

### Server Flow
```
Video Replayer → Format Converter → LSTM Inference → Postprocessor
                                                           ↓
                                                      HolovizOp
                                                   (Render overlays)
                                                           ↓
                                        ┌──────────────────┴─────────┐
                                        ↓                            ↓
                                (Local Display)                UCXX Sender
                                                          (Broadcast @tag=1)
```

### Client Flow
```
UCXX Receiver (@tag=1) → HolovizOp (Display)
  (rendered frame)        (simple display)
```

### Message Flow

1. **Server Processing & Rendering**:
   - Replays surgical video frame
   - Runs LSTM inference for tool detection
   - Post-processes to generate masks and coordinates
   - **Renders visualization with tool tracking overlays using HolovizOp**
   - Displays rendered frame locally
   - Broadcasts **pre-rendered RGBA frame** via UCXX to all connected clients

2. **Client Reception**:
   - Receives **rendered frame** (RGBA image) via UCXX
   - Deserializes image data
   - Displays with HolovizOp (simple image display, no overlay processing needed)

## Multi-Machine Deployment

### Server Machine

```bash
#!/bin/bash
# start_server.sh

export HOLOSCAN_INPUT_PATH=/data/endoscopy

./ucxx_endoscopy_tool_tracking \
  --mode server \
  --hostname 0.0.0.0 \
  --port 50008 \
  --data ${HOLOSCAN_INPUT_PATH}

echo "Server listening on all interfaces, port 50008"
```

### Client Machine

```bash
#!/bin/bash
# start_client.sh

SERVER_IP="192.168.1.100"  # Replace with actual server IP

./ucxx_endoscopy_tool_tracking \
  --mode client \
  --hostname ${SERVER_IP} \
  --port 50008

echo "Client connected to ${SERVER_IP}:50008"
```

## Configuration Details

### UCXX Communication

**Server (Listen Mode):**
```cpp
auto ucxx_endpoint = make_resource<UcxxEndpoint>(
    "ucxx_endpoint",
    Arg("hostname", "0.0.0.0"),  // Listen on all interfaces
    Arg("port", 50008),
    Arg("listen", true)           // Server mode
);

auto holoviz = make_operator<HolovizOp>(
    "holoviz",
    Arg("enable_render_buffer_output") = true,  // Enable output
    Arg("allocator") = allocator
);

auto ucxx_sender = make_operator<UcxxSenderOp>(
    "ucxx_sender",
    Arg("tag", 1ul),              // Message tag
    Arg("endpoint", ucxx_endpoint)
);

// Send rendered frames
add_flow(holoviz, ucxx_sender, {{"render_buffer_output", "in"}});
```

**Client (Connect Mode):**
```cpp
auto ucxx_endpoint = make_resource<UcxxEndpoint>(
    "ucxx_endpoint",
    Arg("hostname", "192.168.1.100"),  // Server IP
    Arg("port", 50008),
    Arg("listen", false)                // Client mode
);

auto ucxx_receiver = make_operator<UcxxReceiverOp>(
    "ucxx_receiver",
    Arg("tag", 1ul),                    // Must match sender tag
    Arg("schema_name", "isaac.Tensor"), // Receiving Tensor (rendered frames)
    Arg("buffer_size", buffer_size),
    Arg("endpoint", ucxx_endpoint)
);
```

### Message Tags

- **Tag 1**: Rendered RGBA frames (Server → Client)

Tags must match between sender and receiver pairs.

### Data Schema

The server broadcasts `isaac.Tensor` (rendered frames) which includes:
- **Schema**: `isaac.Tensor` (defined in `tensor.fbs`)
- **Shape**: [480, 854, 4] (height, width, RGBA channels)
- **Data Type**: uint8 (RGBA8 format, 4 bytes per pixel)
- **Device**: GPU or CPU tensor depending on HolovizOp output
- **Size**: ~1.6 MB per frame (data payload)
- **Content**: Pre-rendered video with all tool tracking overlays

The Tensor schema includes:
```cpp
table Tensor (native_type: "holoscan::Tensor") {
  data: [ubyte];      // Raw pixel data
  shape: [int64];     // [height, width, channels]
  dtype: DLDataType;  // Data type info (uint8)
  device: DLDevice;   // Device context (CPU/GPU)
  ndim: uint32;       // Number of dimensions (3)
  strides: [int64];   // Memory strides
}
```

Serialized using FlatBuffers for efficient transmission with native `holoscan::Tensor` support.

**Key Benefit**: Clients don't need to understand tool tracking data structures - they just display the rendered Tensor as an image!

## Command Reference

### Full Options

```
Usage: ucxx_endoscopy_tool_tracking [options]

Options:
  -d, --data <path>        Path to data directory (required for server)
  -c, --config <path>      Path to config file (optional)
  -h, --hostname <host>    Hostname (default: 0.0.0.0 for server, 127.0.0.1 for client)
  -p, --port <port>        Port number (default: 50008)
  -m, --mode <mode>        Mode: 'server' or 'client' (required)
  -?, --help               Show this help message
```

### Environment Variables

- `HOLOSCAN_INPUT_PATH`: Data directory (server only)
- `HOLOSCAN_CONFIG_PATH`: Custom config file path

## Network Setup

### Firewall Configuration

**Server:**
```bash
# Ubuntu/Debian
sudo ufw allow 50008/tcp

# RHEL/CentOS
sudo firewall-cmd --add-port=50008/tcp --permanent
sudo firewall-cmd --reload
```

**Client:**
No firewall changes needed (outbound connection).

### Testing Connectivity

```bash
# From client machine
nc -zv SERVER_IP 50008

# Or using telnet
telnet SERVER_IP 50008
```

## Performance Optimization

### Server Optimization

1. **GPU Selection**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./ucxx_endoscopy_tool_tracking --mode server ...
   ```

2. **TensorRT FP16** (already enabled):
   - Faster inference with minimal accuracy loss
   - Configured in `ucxx_endoscopy_tool_tracking.yaml`

3. **Buffer Tuning**:
   - Adjust `BlockMemoryPool` sizes in config
   - Monitor GPU memory with `nvidia-smi`

### Network Optimization

1. **Enable RDMA** (if available):
   - UCX automatically detects and uses RDMA
   - Significantly reduces latency

2. **Network Bandwidth**:
   - Test with: `iperf3 -s` (server) and `iperf3 -c SERVER_IP` (client)
   - Expect ~100-500 Mbps for smooth streaming

### Client Optimization

1. **Headless Mode** (no display):
   Modify `holoviz_client.headless: true` in config

2. **Multiple Clients**:
   - Each client connects independently
   - Server broadcasts to all connected clients

## Troubleshooting

### Server Issues

**Problem: "Failed to load model"**
```bash
# Verify data directory
ls $HOLOSCAN_INPUT_PATH/tool_loc_convlstm.onnx

# Check environment
echo $HOLOSCAN_INPUT_PATH
```

**Problem: "Port already in use"**
```bash
# Find process using port
netstat -tulpn | grep 50008

# Use different port
./ucxx_endoscopy_tool_tracking --mode server --port 50009 ...
```

### Client Issues

**Problem: "Connection refused"**
```bash
# 1. Verify server is running
# 2. Test network connectivity
ping SERVER_IP

# 3. Check firewall
telnet SERVER_IP 50008

# 4. Verify hostname
./ucxx_endoscopy_tool_tracking --mode client --hostname SERVER_IP --port 50008
```

**Problem: "Endpoint closed with error"**
- Server may have crashed or stopped
- Network connection interrupted
- Restart both server and client

### Performance Issues

**Low FPS:**
```bash
# Monitor GPU
nvidia-smi -l 1

# Check CPU
htop

# Monitor network
iftop
```

**High Latency:**
- Check network latency: `ping SERVER_IP`
- Reduce visualization quality if needed
- Ensure server GPU has enough memory

## Development Tips

### Adding Custom Processing

**Server-side (before broadcast):**
```cpp
// In server.cpp
auto custom_op = make_operator<MyCustomOp>("custom");
add_flow(tool_tracking_postprocessor, custom_op);
add_flow(custom_op, ucxx_sender);
```

**Client-side (after receive):**
```cpp
// In client.cpp
auto custom_op = make_operator<MyCustomOp>("custom");
add_flow(ucxx_receiver, custom_op);
add_flow(custom_op, holoviz);
```

### Debugging

**Enable verbose logging:**
```bash
HOLOSCAN_LOG_LEVEL=DEBUG ./ucxx_endoscopy_tool_tracking --mode server ...
```

**Monitor UCXX:**
- Check UCX_LOG_LEVEL environment variable
- Monitor connection state in logs

### Testing

**Server without client:**
Server runs independently and shows local visualization even without clients.

**Multiple clients:**
Start multiple client instances pointing to same server.

## Use Case Examples

### 1. Remote Surgical Monitoring
```bash
# OR room: Run server with live camera
# Observation room: Run client to view

# Server (OR)
./ucxx_endoscopy_tool_tracking --mode server --data /live/camera

# Client (Remote)
./ucxx_endoscopy_tool_tracking --mode client --hostname OR_ROOM_IP
```

### 2. Training/Education
```bash
# Instructor station: Run server
# Student stations: Multiple clients

# Instructor
./ucxx_endoscopy_tool_tracking --mode server --data /training/videos

# Each student
./ucxx_endoscopy_tool_tracking --mode client --hostname INSTRUCTOR_IP
```

### 3. Development/Testing
```bash
# Development machine: Run server with test data
# Test machine: Run client for UI testing

# Dev machine
./ucxx_endoscopy_tool_tracking --mode server --data /test/data

# Test machine
./ucxx_endoscopy_tool_tracking --mode client --hostname localhost
```

## Next Steps

1. **Customize Visualization**: Modify `holoviz` and `holoviz_client` configs
2. **Add Recording**: Integrate `VideoStreamRecorderOp` on server or client
3. **Multi-Stream**: Extend to handle multiple video sources
4. **Load Balancing**: Distribute clients across multiple servers

## Reference

- Main README: `README.md`
- Config file: `ucxx_endoscopy_tool_tracking.yaml`
- Server code: `src/server.cpp`, `src/server.h`
- Client code: `src/client.cpp`, `src/client.h`
- Entry point: `src/main.cpp`

## Support

For issues or questions:
- Check logs for error messages
- Verify network connectivity
- Ensure compatible versions of UCX and Holoscan SDK
- Consult [Holoscan Documentation](https://docs.nvidia.com/holoscan/)
