# Video Streaming Demo Enhanced

This unified application demonstrates how to use the Holoscan SDK to create both streaming client and server applications for bidirectional video communication. The demo includes both client and server components that can work together to showcase real-time video streaming capabilities.

## Overview

The video streaming demo provides:
- **Streaming Client**: Captures video from V4L2 cameras or video files and streams to a server
- **Streaming Server**: Receives video streams from clients and can echo them back (passthrough mode)
- **Bidirectional Communication**: Both sending and receiving video frames
- **Multiple Source Support**: V4L2 cameras, video replay files

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher  
- Holoscan SDK 3.5.0 or higher
- V4L2 camera (optional, for live streaming)

## Dependencies

### Client Dependencies

Download the client streaming binaries from NGC:

```bash
# Navigate to the client operator directory
cd <your_holohub_path>/operators/video_streaming/streaming_client_enhanced

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up
rm -rf holoscan_client_cloud_streaming_v0.2
```

### Server Dependencies

Download the server streaming binaries from NGC:

```bash
# Navigate to the server operator directory  
cd <your_holohub_path>/operators/video_streaming/streaming_server_enhanced

# Download using NGC CLI
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.2"
unzip -o holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming

# Clean up
rm -rf holoscan_server_cloud_streaming_v0.2
```

## Running the Applications

The unified application provides both client and server as separate components:

**⚠️ Important: Enhanced applications require Holoscan SDK 3.5.0. Use the base image parameter to ensure compatibility.**

### 1. Start the Streaming Server

```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp
```

### 2. Start the Streaming Client (in another terminal)

**Option A: V4L2 Camera (Webcam)**
```bash
# From holohub root directory - captures live video from webcam
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp
```

**Option B: Video File Replay**
```bash
# From holohub root directory - replays pre-recorded video file
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp --run-args='-c streaming_client_demo_replayer.yaml'
```

**Key Differences:**
- **V4L2 Mode**: Uses `streaming_client_demo.yaml` (default), captures from webcam, 640x480 resolution
- **Replayer Mode**: Uses `streaming_client_demo_replayer.yaml` (custom), plays video file, 854x480 resolution

### Backward Compatibility

The original separate applications are still available for backward compatibility:

```bash
# Original applications (still work) - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu streaming_server_demo_enhanced --language cpp
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu streaming_client_demo_enhanced --language cpp
```

## Command Line Options

### Server Options
- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_server_demo.yaml)
- `-d, --data <directory>`: Data directory for video files

### Client Options  
- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_client_demo.yaml)
- `-d, --data <directory>`: Data directory for video files

**Note:** Video source type (V4L2 vs replayer) is configured in the YAML file, not via command line arguments.

## Configuration

### Server Configuration (streaming_server_demo.yaml)

```yaml
# Streaming server settings
streaming_server:
  # Video/stream parameters
  width: 1280          # Frame width in pixels
  height: 720          # Frame height in pixels  
  fps: 30              # Frame rate
  
  # Server connection settings
  server_ip: "127.0.0.1"
  port: 48010          # Streaming port
  multi_instance: false
  server_name: "VideoStreamingServer"
  
  # Operation mode
  receive_frames: true
  send_frames: true
  visualize_frames: false
```

### Client Configuration

#### V4L2 Camera Configuration (streaming_client_demo.yaml)

```yaml
# Source configuration - V4L2 camera mode
source: "v4l2"

# Streaming client settings
streaming_client:
  width: 640           # V4L2 camera resolution
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  visualize_frames: true

# V4L2 camera configuration
v4l2_source:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Format converter - V4L2 outputs RGBA8888 (4 channels)
format_converter:
  in_dtype: "rgba8888"         # V4L2 always outputs RGBA8888
  out_dtype: "rgb888"          # Convert to RGB888 (3 channels)
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0] # Convert RGB to BGR
```

#### Video Replayer Configuration (streaming_client_demo_replayer.yaml)

```yaml
# Source configuration - Video file replay mode
source: "replayer"

# Streaming client settings
streaming_client:
  width: 854           # Video file resolution (matches surgical_video.gxf)
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  visualize_frames: true

# Video replayer configuration
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30
  repeat: true
  realtime: true
  count: 0

# Format converter - Video replayer outputs RGB888 (3 channels)
format_converter:
  in_dtype: "rgb888"           # Video replayer outputs RGB888
  out_dtype: "rgb888"          # Keep as RGB888 (3 channels)
  out_tensor_name: tensor
  scale_min: 0.0
  scale_max: 255.0
  out_channel_order: [2, 1, 0] # Convert RGB to BGR
```

## Camera Setup and Testing

### Testing Your V4L2 Camera

Before starting the streaming client with camera input:

```bash
# Check available video devices
ls -la /dev/video*

# Get camera information
v4l2-ctl --device=/dev/video0 --info

# Test camera with recommended resolution
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-count=10

# List supported formats
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### Recommended Resolution Settings

**For V4L2 cameras (like Logitech C920):**
- **1280x720 @ 30fps** - Best balance of quality and performance
- **1920x1080 @ 30fps** - High quality streaming (if supported)
- **854x480 @ 30fps** - Default, good for testing and lower bandwidth

**Important:** Ensure both client and server use matching resolution settings for optimal performance.

## Workflow Examples

### Example 1: Local Camera to Server Echo (Unified Application)

1. **Build Application:**
   ```bash
   ./holohub build video_streaming_demo_enhanced --language cpp
   ```

2. **Start Server (Terminal 1):**
   ```bash
   ./holohub run --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced --language cpp
   ```

3. **Start Client with Camera (Terminal 2):**
   ```bash
   cd build/video_streaming_demo_enhanced/applications/video_streaming_demo_enhanced/cpp
   docker run --net host --runtime nvidia --gpus all -v /dev:/dev -e device=/dev/video0 -v $(pwd):/workspace/app -w /workspace/app holohub:video_streaming_demo_enhanced ./streaming_client_demo_enhanced
   ```

4. **Expected Behavior:**
   - Client captures video from `/dev/video0`
   - Client sends frames to server
   - Server receives frames and echoes them back
   - Client displays received frames (if `visualize_frames: true`)

### Example 2: Video File Replay (Unified Application)

1. **Build Application:**
   ```bash
   ./holohub build video_streaming_demo_enhanced --language cpp
   ```

2. **Start Server (Terminal 1):**
   ```bash
   ./holohub run --docker-opts='-e EnableHybridMode=1' video_streaming_demo_enhanced --language cpp
   ```

3. **Start Client with Video File (Terminal 2):**
   ```bash
   cd build/video_streaming_demo_enhanced/applications/video_streaming_demo_enhanced/cpp
   docker run --net host --runtime nvidia --gpus all -v $(pwd):/workspace/app -v /home/cdinea/Downloads/enhancedapp_holohub/holohub/data:/workspace/holohub/data -w /workspace/app holohub:video_streaming_demo_enhanced ./streaming_client_demo_enhanced -c streaming_client_demo_replayer.yaml
   ```

4. **Expected Behavior:**
   - Client replays video from `/workspace/holohub/data/endoscopy/surgical_video.gxf`
   - Client sends frames to server (854x480 resolution)
   - Server receives frames and echoes them back
   - Client displays received frames (if `visualize_frames: true`)


## Video Source Modes

### V4L2 Camera Mode vs Video Replayer Mode

| Feature | V4L2 Camera | Video Replayer |
|---------|-------------|----------------|
| **Config File** | `streaming_client_demo.yaml` (default) | `streaming_client_demo_replayer.yaml` (custom) |
| **Command** | `--docker-opts='-e device=/dev/video0'` | `--run-args='-c streaming_client_demo_replayer.yaml'` |
| **Source Type** | `source: "v4l2"` | `source: "replayer"` |
| **Input Format** | `rgba8888` (4 channels) | `rgb888` (3 channels) |
| **Resolution** | 640x480 | 854x480 |
| **Data Source** | Live webcam | Pre-recorded surgical video |
| **Use Case** | Real-time streaming | Testing, demos, development |

### Switching Between Modes

To switch between V4L2 camera and video replayer:

1. **Stop the current client** (Ctrl+C)
2. **Use the appropriate command:**
   - For camera: `./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' streaming_client_demo_enhanced --language cpp`
   - For video replay: `./holohub run --docker-opts='-e EnableHybridMode=1' streaming_client_demo_enhanced --language cpp --run-args='-c streaming_client_demo_replayer.yaml'`

**Important:** The server doesn't need to be restarted when switching client modes.

## Troubleshooting

### Camera Issues
- **Camera not detected:**
  ```bash
  sudo usermod -a -G video $USER
  # Log out and back in, then test again
  ```

- **Permission denied:**
  ```bash
  sudo chmod 666 /dev/video0
  ```

### Performance Issues
- **Poor streaming quality:**
  - Try lower resolution (854x480 or 640x480)
  - Reduce frame rate to 15 or 24 FPS
  - Ensure client and server resolutions match

### Connection Issues
- **Server not starting:**
  ```bash
  # Check if port is already in use
  netstat -tlnp | grep 48010
  
  # Kill existing process if needed
  sudo lsof -ti:48010 | xargs sudo kill -9
  ```

- **Client connection timeout:**
  - Verify server is running first
  - Check firewall settings for port 48010
  - Ensure server_ip and port match in both configurations

### Video Replayer Issues

- **Config file not found:**
  ```bash
  # Ensure the replayer config exists in build directory
  cp applications/streaming_client_demo_enhanced/cpp/streaming_client_demo_replayer.yaml build/streaming_client_demo_enhanced/
  ```

- **Format converter errors:**
  - `Invalid channel count for RGBA8888 3 != 4`: Video replayer outputs RGB888 (3 channels), not RGBA8888 (4 channels)
  - Solution: Use `streaming_client_demo_replayer.yaml` which has correct format converter settings

- **Resolution mismatch:**
  - Video file is 854x480, ensure all components use matching resolution
  - Check `streaming_client`, `holoviz`, and `format_converter` settings

### Expected Behavior and Logs

**Client Application:** 
The streaming client may show `GXF_EXCEEDING_PREALLOCATED_SIZE` errors during BGR→BGRA conversion. This is expected behavior as the operators handle dynamic buffer allocation internally.

**Server Application:**
The server should display connection status and frame processing information. Look for messages about client connections and frame throughput.

**Successful Video Replayer Logs:**
```
[info] Source set to: replayer
[info] Using video replayer as source
[info] Connection established successfully
[info] Tensor validation passed: 480x854x3, 1229760 bytes
[info] Frame sent successfully
```

## Directory Structure

```
video_streaming_demo_enhanced/
├── client/                          # Client application components
│   ├── cpp/                        # C++ client implementation
│   │   ├── streaming_client_demo_enhanced    # Built client executable
│   │   ├── streaming_client_demo.cpp         # Client source code
│   │   ├── streaming_client_demo.yaml        # Client V4L2 config
│   │   ├── streaming_client_demo_replayer.yaml # Client replayer config
│   │   └── CMakeLists.txt                     # Client build configuration
│   └── setup_streaming_client.sh   # Client setup script
├── server/                          # Server application components
│   ├── cpp/                        # C++ server implementation
│   │   ├── streaming_server_demo_enhanced    # Built server executable
│   │   ├── streaming_server_demo.cpp         # Server source code
│   │   ├── streaming_server_demo.yaml        # Server config
│   │   └── CMakeLists.txt                     # Server build configuration
│   └── setup_streaming_server.sh   # Server setup script
├── CMakeLists.txt                   # Main build configuration with add_holohub_application
├── metadata.json                    # Application metadata
├── Dockerfile                       # Container configuration
└── README.md                        # This file
```

**Key Features:**
- **Unified build**: Single `./holohub build` command builds both client and server
- **Clean structure**: Maintains separate client and server directories
- **HoloHub integration**: Uses `add_holohub_application` for proper CLI integration
- **Backward compatibility**: Works with existing separate application commands

## Integration Testing

The video streaming demo includes comprehensive integration testing to verify end-to-end functionality between client and server components.

### Integration Test Overview

The integration test validates:
- **Server Startup**: Streaming server initializes and starts listening
- **Client Connection**: Streaming client connects to server successfully  
- **Video Streaming**: Bidirectional video frame transmission (client→server→client)
- **Resource Management**: Proper cleanup and resource handling
- **Error Handling**: Graceful handling of connection issues

### Running Integration Tests

#### Automated Integration Test (Recommended)

The integration test script (`integration_test.sh`) runs the complete end-to-end test in a Docker container with proper SDK version and dependencies.

```bash
# From the video_streaming_demo_enhanced directory
cd applications/video_streaming_demo_enhanced
./integration_test.sh
```

**OR from holohub root:**
```bash
./applications/video_streaming_demo_enhanced/integration_test.sh
```

**Test Configuration:**
- **Duration**: 3-5 minutes total (includes Docker build and test execution)
- **SDK Version**: Holoscan 3.5.0 (enforced via environment variable)
- **Test Duration**: 30 seconds of active streaming
- **Requirements**: Docker, NVIDIA GPU, committed C++ source code
- **Output**: Detailed logs saved to `integration_test.log`

**⚠️ Important Notes:**
1. The test runs in Docker and builds from **committed source code**
2. If you have local C++ changes, **commit them first** before running the test
3. The test uses cached Docker layers for faster builds (unless cache is cleared)

#### Manual Integration Test

For manual testing and debugging:

```bash
# Terminal 1: Start Server
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced server --language cpp

# Terminal 2: Start Client (after server is running)
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced client_replayer --language cpp
```

### Integration Test Process

The automated integration test (`integration_test.sh`) follows this sequence:

#### 1. **Pre-Test Setup** (10-20 seconds)
```bash
# Displays current git commit
echo "Current commit: $(git log --oneline -1)"

# Cleans Docker build cache (optional, for fresh builds)
docker system prune -f --filter "label=holohub"

# Sets SDK version environment variable
export HOLOHUB_BASE_SDK_VERSION=3.5.0
```

#### 2. **Docker Build & Test Execution** (2-4 minutes)
```bash
# Builds Docker image and runs CTest
./holohub test video_streaming_demo_enhanced \
  --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu \
  --cmake-options="-DBUILD_TESTING=ON" \
  --ctest-options="-R video_streaming_integration_test -V" \
  --verbose
```

**What happens internally:**
- Builds Docker image with Holoscan SDK 3.5.0
- Compiles server and client C++ applications
- Copies configuration files to build directory
- Runs CTest with the integration test

#### 3. **Integration Test Execution** (54 seconds)
The `video_streaming_integration_test` defined in CMakeLists.txt:

1. **Server Startup** (10 seconds)
   - Launches streaming server in background: `streaming_server_demo_enhanced`
   - Uses config: `streaming_server_demo.yaml`
   - Waits for initialization (15 stability checks over 30 seconds)

2. **Client Connection & Streaming** (30 seconds)
   - Starts streaming client: `streaming_client_demo_enhanced`
   - Uses config: `streaming_client_demo_replayer.yaml` (video replay mode)
   - Establishes connection to server
   - Streams video frames bidirectionally for 30 seconds

3. **Verification & Cleanup**
   - Verifies both processes are still running (connection successful)
   - Gracefully terminates client (SIGTERM)
   - Gracefully terminates server (SIGTERM)
   - Reports PASS/FAIL based on process stability

#### 4. **Post-Test Analysis** (5 seconds)
```bash
# Verifies test results from log file
if grep -q "Test.*Passed\|100%.*tests passed" integration_test.log; then
  echo "✓ Integration test PASSED"
  exit 0
fi
```

### Success Criteria

The integration test **PASSES** when **ALL** conditions are met:

#### ✅ Server Success Criteria
1. **Process Stability**: Server process runs continuously for entire test (54+ seconds)
2. **Initialization**: Server starts and becomes stable within 30 seconds
3. **Client Connection**: Successfully accepts client connection
4. **Frame Reception**: Receives frames from client (logged in verbose output)
5. **Frame Transmission**: Sends frames back to client (bidirectional communication)
6. **Graceful Shutdown**: Stops cleanly without crashes

#### ✅ Client Success Criteria
1. **Process Stability**: Client process runs continuously for 30+ seconds
2. **Server Connection**: Successfully connects to server on port 48010
3. **Video Replay**: Successfully loads and plays surgical video file
4. **Frame Transmission**: Sends frames to server (500+ frames in 30 seconds)
5. **Frame Reception**: Receives frames back from server (bidirectional communication)
6. **Graceful Shutdown**: Stops cleanly without crashes

#### ✅ CTest Success Criteria
1. **Test Execution**: `video_streaming_integration_test` completes without timeout
2. **Test Status**: CTest reports test as "Passed"
3. **Exit Code**: Overall test exit code is 0
4. **Log Output**: Contains "✓ Integration test PASSED" message

### Expected Output

#### Console Output (Successful Test)
```bash
=== Video Streaming Demo Integration Test ===
This test may take up to 10 minutes to complete...
NOTE: Test runs in Docker and uses committed source code (not local build)

Forcing Docker to use latest committed changes...
Current commit: 72240492 Run only integration test, skip individual client/server tests

Cleaning Docker build cache...
Total reclaimed space: 0B

Running integration test with Docker (using committed fixes)...

[Docker build output...]
Step 1/15 : ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
Step 2/15 : FROM ${BASE_IMAGE}
[...]

[CTest output...]
UpdateCTestConfiguration from :/workspace/holohub/build/DartConfiguration.tcl
Test project /workspace/holohub/build
    Start 1: video_streaming_integration_test

1: === Custom Integration Test ===
1: Starting server and client in same container...
1: Starting streaming server...
1: Waiting for server to initialize...
1: Checking if server is running and stable...
1: Server process is running (attempt 1/15)
1: Server process is running (attempt 2/15)
[...]
1: Server process is running (attempt 5/15)
1: ✓ Server is running and stable
1: ✓ Server started successfully
1: Starting streaming client...
1: Letting client stream for 30 seconds...
[Streaming logs showing frame transmission...]
1: [info] Frame sent successfully
1: [info] CLIENT: Received frame #533 from server: 854x480
1: ✓ Client is still running - connection successful
1: Stopping client...
1: Cleaning up server process...
1: ✓ Integration test PASSED

1/1 Test #1: video_streaming_integration_test ...   Passed   54.05 sec

The following tests passed:
	video_streaming_integration_test

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 54.06 sec

=== VERIFICATION ===
✓ Integration test passed with detailed verification
✓ Server component verified
✓ Client component verified
✓ Integration test PASSED
```

#### Key Log Patterns to Look For

**Server Success Indicators:**
```
[info] StreamingServerResource starting...
[info] StreamingServerUpstreamOp::start() called
[info] StreamingServerDownstreamOp::start() called
[info] ✅ UPSTREAM: Client connected successfully
[info] ✅ Processing UNIQUE frame: 854x480, 1639680 bytes
[info] ✅ DOWNSTREAM: Frame sent successfully to StreamingServerResource
```

**Client Success Indicators:**
```
[info] Source set to: replayer
[info] Using video replayer as source
[info] StreamingClient created successfully
[info] Connection established successfully
[info] ✅ Tensor validation passed: 480x854x3, 1229760 bytes
[info] ✅ Frame sent successfully on attempt 1
[info] 🎯 CLIENT: Frame received callback triggered!
[info] 📥 CLIENT: Received frame #533 from server: 854x480
```

**Performance Indicators:**
```
# Server processed ~565 frames
[info] 📊 DOWNSTREAM: Processing tensor 565 - shape: 480x854x4

# Client sent and received ~533 frames
[info] 📥 CLIENT: Received frame #533 from server

# Frame rate: ~17-19 FPS (533 frames ÷ 30 seconds)
```

#### Integration Test Log File

The complete test execution is saved to `integration_test.log` (typically 25,000-30,000 lines). This file contains:

1. **Docker Build Logs**: Complete build output with all dependencies
2. **CMake Configuration**: Build configuration and test setup
3. **CTest Execution**: Detailed test execution with timestamps
4. **Server Logs**: All server application logs (initialization, frame processing, shutdown)
5. **Client Logs**: All client application logs (connection, streaming, frame reception)
6. **Test Summary**: Final PASS/FAIL status with verification details

**Analyzing the log:**
```bash
# Check test status
grep "Integration test PASSED" integration_test.log

# Check frame counts
grep "CLIENT: Received frame" integration_test.log | tail -5

# Check for errors
grep -i "error\|fail\|crash" integration_test.log

# View test summary
tail -100 integration_test.log
```

### Troubleshooting Integration Tests

#### Common Issues and Solutions

**Build Failures:**
```bash
# Clean build and retry
rm -rf build/
./holohub build video_streaming_demo_enhanced --language cpp
```

**Server Connection Issues:**
```bash
# Check if port is in use
netstat -tlnp | grep 48010
sudo lsof -ti:48010 | xargs sudo kill -9
```

**Client Connection Timeout:**
- Verify server started successfully (check server logs)
- Ensure firewall allows port 48010
- Check Docker network connectivity

**Frame Transmission Issues:**
- Verify video data files exist: `/workspace/holohub/data/endoscopy/`
- Check format converter settings in config files
- Monitor GPU memory usage

#### Debug Mode

For detailed debugging, run components separately:

```bash
# Server with verbose logging
HOLOSCAN_LOG_LEVEL=DEBUG ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced --language cpp

# Client with verbose logging  
HOLOSCAN_LOG_LEVEL=DEBUG ./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_enhanced --language cpp --run-args='-c streaming_client_demo_replayer.yaml'
```

### Integration Test Files

The integration test generates one comprehensive log file:

- **`integration_test.log`**: Complete test execution log containing:
  - Docker build output with all dependencies
  - CMake configuration and build logs
  - CTest execution with detailed timestamps
  - Server application logs (initialization, frame processing, shutdown)
  - Client application logs (connection, streaming, frame reception)
  - Test summary with final PASS/FAIL status

This file contains all information needed for debugging failed tests (~25,000-30,000 lines for a complete run).

### Continuous Integration

The integration test is designed for CI/CD pipelines:

```bash
# CI-friendly command with timeout and exit codes
timeout 300 ./applications/video_streaming_demo_enhanced/integration_test.sh
echo "Integration test exit code: $?"
```

**Exit Codes:**
- `0`: All tests passed successfully
- `1`: Test failures detected
- `124`: Test timeout (5 minutes)

## Operator Documentation

For detailed information about the underlying video streaming operators used in this application, see:

📋 **[Video Streaming Operators](../../operators/video_streaming/README.md)** - Complete operator documentation

The operator documentation includes:
- **Client Components**: StreamingClientOp, FrameSaverOp
- **Server Components**: StreamingServerResource, StreamingServerUpstreamOp, StreamingServerDownstreamOp
- **Parameters and Configuration**: Detailed parameter descriptions and usage examples
- **Testing Documentation**: Comprehensive test suite with 40+ tests passing
- **API Reference**: Complete API documentation for all components

## Performance Notes

- **GPU Memory**: Configure appropriate allocator block sizes for your resolution
- **Network Bandwidth**: Monitor bandwidth usage for remote streaming scenarios  
- **Frame Rate**: Higher frame rates require more GPU/CPU resources
- **Resolution**: Balance between quality and performance based on your hardware

## License

Apache-2.0 - See the LICENSE file for details.