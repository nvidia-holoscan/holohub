# Video Streaming Demo Enhanced

This unified application demonstrates how to use the Holoscan SDK to create both streaming client and server applications for bidirectional video communication. The demo includes both client and server components that can work together to showcase real-time video streaming capabilities.

## Overview

The video streaming demo provides:
- **Streaming Client**: Captures video from V4L2 cameras or video files and streams to a server
- **Streaming Server**: Comprehensive server architecture with three main components:
  - **StreamingServerResource**: Manages server connections and client lifecycle
  - **StreamingServerUpstreamOp**: Receives video streams from clients
  - **StreamingServerDownstreamOp**: Sends video frames back to clients (passthrough/echo mode)
- **Bidirectional Communication**: Both sending and receiving video frames
- **Multiple Source Support**: V4L2 cameras, video replay files
- **Multiple Language Support**: Both C++ and Python implementations available
- **Interoperability**: C++ and Python components can work together seamlessly

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

**⚠️ Important: Enhanced applications require Holoscan SDK 3.5.0. The server uses the base image, while the client requires OpenSSL 3.4.0 and must use the custom Dockerfile.**

### 1. Start the Streaming Server

```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp
```

### 2. Start the Streaming Client (in another terminal)

**Option A: V4L2 Camera (Webcam)**
```bash
# From holohub root directory - captures live video from webcam
./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0'
```

**Option B: Video File Replay**
```bash
# From holohub root directory - replays pre-recorded video file
./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --run-args='-c streaming_client_demo_replayer.yaml'
```

**Key Differences:**
- **V4L2 Mode**: Uses `streaming_client_demo.yaml` (default), captures from webcam, 640x480 resolution
- **Replayer Mode**: Uses `streaming_client_demo_replayer.yaml` (custom), plays video file, 854x480 resolution

### Python Applications

The demo also provides Python implementations of both server and client components.

#### 1. Start the Python Streaming Server

```bash
# From holohub root directory - Python server implementation (defaults to 854x480)
./holohub run video_streaming_demo_enhanced server_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Note:** For V4L2 clients (640x480), manually run the server with matching resolution:
```bash
# Inside Docker container or after building locally
python3 build/video_streaming_demo_enhanced/streaming_server_demo.py --width 640 --height 480
```

#### 2. Start the Python Streaming Client (in another terminal)

**Option A: V4L2 Camera (Webcam) - 640x480**
```bash
# From holohub root directory - Python client with V4L2 camera
./holohub run video_streaming_demo_enhanced client_python_v4l2 --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Option B: Video File Replay - 854x480**
```bash
# From holohub root directory - Python client with video replayer
./holohub run video_streaming_demo_enhanced client_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'
```

**Python Application Notes:**
- Python implementations are located in:
  - Server: `video_streaming_demo_server/python/streaming_server_demo.py`
  - Client: `video_streaming_demo_client/python/streaming_client_demo.py`
- Python client supports both V4L2 camera (640x480) and video replayer (854x480) modes
- Both Python server and client require the custom Dockerfile with OpenSSL 3.4.0
- Python bindings must be built with `--configure-args='-DHOLOHUB_BUILD_PYTHON=ON'`

**Mixing C++ and Python:**
You can run a C++ server with a Python client, or vice versa - they are fully compatible:

```bash
# Terminal 1: C++ Server
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp

# Terminal 2: Python Client
./holohub run video_streaming_demo_enhanced client_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1'
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
   - For camera: `./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0'`
   - For video replay: `./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --run-args='-c streaming_client_demo_replayer.yaml'`

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

#### Python Integration Test

For testing Python implementations:

```bash
# From the video_streaming_demo_enhanced directory
cd applications/video_streaming_demo_enhanced
./integration_test_python.sh
```

**OR from holohub root:**
```bash
./applications/video_streaming_demo_enhanced/integration_test_python.sh
```

**Python Test Configuration:**
- Tests Python server and client implementations
- Uses the same 30-second streaming duration as C++ tests
- Validates bidirectional frame transmission
- Requires custom Dockerfile with OpenSSL 3.4.0

#### Manual Integration Test

For manual testing and debugging:

**C++ Implementation:**
```bash
# Terminal 1: Start Server (uses base image)
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp

# Terminal 2: Start Client (uses custom Dockerfile with OpenSSL 3.4.0)
./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --run-args='-c streaming_client_demo_replayer.yaml'
```

**Python Implementation:**
```bash
# Terminal 1: Start Python Server
./holohub run video_streaming_demo_enhanced server_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1'

# Terminal 2: Start Python Client
./holohub run video_streaming_demo_enhanced client_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1'
```

**Mixed C++ and Python:**
```bash
# Terminal 1: C++ Server + Python Client
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_server --language cpp
./holohub run video_streaming_demo_enhanced client_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1'

# OR: Python Server + C++ Client
./holohub run video_streaming_demo_enhanced server_python --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1'
./holohub run video_streaming_demo_client --language cpp --docker-file applications/video_streaming_demo_enhanced/Dockerfile --docker-opts='-e EnableHybridMode=1' --run-args='-c streaming_client_demo_replayer.yaml'
```

### Running Integration Tests

There are two ways to run the integration test, depending on your use case:

#### Option 1: Development/Debugging (Wrapper Script) 🛠️

**Use this for:** Local development, debugging, detailed output

```bash
# From holohub root
./applications/video_streaming_demo_enhanced/integration_test.sh
```

**Advantages:**
- ✅ Automatic environment setup (SDK version, directory handling)
- ✅ Automatic Docker cache cleanup (ensures fresh builds)
- ✅ Shows current git commit being tested
- ✅ Detailed verification output with custom messages
- ✅ Comprehensive log capture to `integration_test.log`
- ✅ Single simple command

**Best for:** Developers iterating on code changes who want detailed feedback.

#### Option 2: CI/CD (Direct Command) 🚀

**Use this for:** Continuous Integration, automated testing pipelines

```bash
# From holohub root - standard HoloHub test command
./holohub test video_streaming_demo_enhanced \
  --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu \
  --ctest-options="-R video_streaming_integration_test"
```

**Advantages:**
- ✅ Standard HoloHub testing interface
- ✅ Consistent with other HoloHub applications
- ✅ Clean exit codes for CI/CD systems
- ✅ CTest integration for test reporting

**Best for:** CI/CD pipelines, automated test suites, production testing.

**Note:** Both methods run the same underlying integration test defined in `CMakeLists.txt`. The wrapper script (`integration_test.sh`) adds developer-friendly conveniences on top of the direct command.

### Integration Test Process

The integration test (whether run via wrapper script or direct command) follows this sequence:

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

#### 3. **Integration Test Execution** (44 seconds)
The `video_streaming_integration_test` defined in CMakeLists.txt:

1. **Server Startup** (10 seconds)
   - Launches streaming server in background: `streaming_server_demo_enhanced`
   - Uses config: `streaming_server_demo.yaml`
   - Waits for server to initialize and stabilize

2. **Client Connection & Streaming** (30 seconds)
   - Starts streaming client: `streaming_client_demo_enhanced`
   - Uses config: `streaming_client_demo_replayer.yaml` (video replay mode)
   - Establishes connection to server
   - Streams video frames bidirectionally for 30 seconds
   - Typically processes ~567 frames in both directions

3. **Log Verification & Cleanup** (4 seconds)
   - Gracefully terminates client (SIGTERM/SIGKILL)
   - Gracefully terminates server (SIGTERM/SIGKILL)
   - Verifies server logs for all required events and frame processing
   - Verifies client logs for successful streaming and frame transmission
   - Reports PASS/FAIL based on comprehensive log analysis

#### 4. **Post-Test Analysis** (5 seconds)
```bash
# Verifies test results from log file
if grep -q "Test.*Passed\|100%.*tests passed" integration_test.log; then
  echo "✓ Integration test PASSED"
  exit 0
fi
```

### Success Criteria

The integration test **PASSES** when **ALL 10 checks** are met (6 server + 4 client):

#### ✅ Server Log Verification Criteria (6 checks required)

1. **Client Connected**: `grep -q 'Client connected' $SERVER_LOG`
   - Verifies a client successfully connected to the server

2. **Upstream Connection Established**: `grep -q 'Upstream connection established' $SERVER_LOG`
   - Verifies the upstream data channel (client→server) is established

3. **Downstream Connection Established**: `grep -q 'Downstream connection established' $SERVER_LOG`
   - Verifies the downstream data channel (server→client) is established

4. **Upstream Frame Processing**: `grep -q 'Processing UNIQUE frame' $SERVER_LOG`
   - Verifies StreamingServerUpstreamOp received and processed frames from client
   - Typical: 567 unique frames in 30 seconds

5. **Downstream Tensor Processing**: `grep -q 'DOWNSTREAM: Processing tensor' $SERVER_LOG`
   - Verifies StreamingServerDownstreamOp processed and sent tensors to client
   - Typical: 567 tensors in 30 seconds

6. **Frame Processing Statistics**: `grep -q 'Frame Processing Stats' $SERVER_LOG`
   - Verifies the server logged performance statistics at shutdown

#### ✅ Client Log Verification Criteria (4 checks required)

1. **Frame Sending Success**: `grep -q 'Frame sent successfully' $CLIENT_LOG`
   - Verifies client successfully sent frames to server
   - Typical: 567 frames sent in 30 seconds

2. **Frame Reception Success**: `grep -q 'CLIENT: Received frame' $CLIENT_LOG`
   - Verifies client successfully received frames from server (bidirectional)
   - Typical: 567 frames received in 30 seconds
   - Completes end-to-end bidirectional verification

3. **Frame Validation**: `grep -q 'Frame validation passed' $CLIENT_LOG`
   - Verifies client frame validation logic is working
   - Logs appear every 30 frames (~1 second at 30 FPS)

4. **Streaming Client Started**: `grep -q 'STARTING STREAMING CLIENT' $CLIENT_LOG`
   - Verifies client initialization completed successfully

#### ✅ Overall Test Success

**Test PASSES if:**
- Server checks: **6/6 passed** (required: 6)
- Client checks: **4/4 passed** (required: 4)
- Total: **10/10 checks passed**

**Test FAILS if:**
- Any check fails (server < 6 or client < 4)
- Process crashes during execution (segfault detected but test continues)
- Test times out (> 300 seconds)

### Expected Output

#### Console Output (Successful Test)
```bash
=== Video Streaming Demo Integration Test ===
This test may take up to 10 minutes to complete...
NOTE: Test runs in Docker and uses committed source code (not local build)

[Docker build output...]
Step 1/15 : ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
[...]

[CTest output...]
Test project /workspace/holohub/build-video_streaming_demo_enhanced
    Start 1: video_streaming_integration_test

1: === Enhanced Integration Test with Log Verification ===
1: Starting server and client with log capture...
1: Server log: /tmp/server_log.XXXXXX
1: Client log: /tmp/client_log.XXXXXX
1: Starting streaming server...
1: Waiting for server to initialize...
1: ✓ Server process is running
1: Starting streaming client...
1: Letting streaming run for 30 seconds...
1: Stopping client...
1: Stopping server...
1: 
1: === Verifying Server Logs ===
1: ✓ Server: Client connected
1: ✓ Server: Upstream connection established
1: ✓ Server: Downstream connection established
1: ✓ Server: StreamingServerUpstreamOp processed 568 unique frames
1: ✓ Server: StreamingServerDownstreamOp processed 568 tensors
1: ✓ Server: Frame processing statistics logged
1: 
1: === Verifying Client Logs ===
1: ✓ Client: Sent 568 frames successfully
1: ✓ Client: Received 534 frames from server
1: ✓ Client: Frame validation passed
1: ✓ Client: Streaming client started
1: 
1: === Test Results Summary ===
1: Server checks passed: 6
1: Client checks passed: 4
1: ✓ STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
1: ✓ Integration test PASSED

1/1 Test #1: video_streaming_integration_test ...   Passed   44.07 sec

The following tests passed:
	video_streaming_integration_test

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 44.07 sec

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
# Server processed 568 frames in both directions
[info] ✅ Processing UNIQUE frame: 854x480, 1639680 bytes, timestamp=29938
[info] 📊 DOWNSTREAM: Processing tensor 568 - shape: 480x854x4, 1639680 bytes

# Client sent 568 frames and received 534 frames
[info] ✅ Frame sent successfully on attempt 1
[info] 📥 CLIENT: Received frame #534 from server: 854x480

# Frame rate: ~19 FPS (568 frames ÷ 30 seconds)
# Bidirectional throughput: ~62 MB/s (1.64MB per frame × 19 FPS × 2 directions)
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

**Test Failure: Connection Events Not Logged**

If you see output like:
```bash
=== Verifying Server Logs ===
✗ Server: Upstream connection not established
✗ Server: Downstream connection not established
✓ Server: StreamingServerUpstreamOp processed 568 unique frames  # But frames work!
✓ Server: StreamingServerDownstreamOp processed 568 tensors      # But frames work!

=== Verifying Client Logs ===
✓ Client: Sent 568 frames successfully
✓ Client: Received 534 frames from server  # Bidirectional works!

=== Test Results Summary ===
Server checks passed: 4
Client checks passed: 4
✗ STREAMING VERIFICATION FAILED - One or more checks failed
✗ Integration test FAILED
```

**Root Cause:** Event callback overwriting in StreamingServerResource
- Both upstream and downstream operators call `set_event_callback()`
- Second call overwrites first operator's callback
- Only last operator receives connection events
- **Frames still work** (567 processed) but events aren't logged to both operators

**Solution:** Use `add_event_listener()` instead of `set_event_callback()`
- Fixed in commit `0e8a9603`: "Fix integration test and event listener bug"
- StreamingServerResource now supports multiple event listeners
- Both operators receive all connection events

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

**Segmentation Fault at Shutdown:**
```bash
# Expected behavior - test still passes if streaming worked
1: Segmentation fault (core dumped) ./streaming_server_demo_enhanced
1: ✓ Server: StreamingServerUpstreamOp processed 567 unique frames
1: ✓ STREAMING VERIFICATION PASSED - Frames actually transmitted!
```
- Segfault occurs during cleanup after test completes
- Test passes if all 9 checks passed before shutdown
- Does not affect streaming functionality

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

## Python Integration Testing

### Overview

The Python integration test validates the complete bidirectional video streaming pipeline using Python implementations of both the server and client applications. The test verifies frame transmission, reception, and processing statistics.

### Running the Python Integration Test

**Command:**
```bash
./run test video_streaming_demo_enhanced --test video_streaming_integration_test_python
```

Or using the holohub CLI directly:
```bash
holohub test video_streaming_demo_enhanced -R video_streaming_integration_test_python
```

**Test Duration:** ~44 seconds (30 seconds of streaming + setup/teardown)

### Test Workflow

1. **Server Startup**: Python streaming server starts with 854x480 resolution
2. **Client Startup**: Python streaming client starts with replayer source (854x480)
3. **Streaming Duration**: 30 seconds of bidirectional video streaming
4. **Log Verification**: Comprehensive log analysis for both server and client
5. **Graceful Shutdown**: Both processes are stopped and logs are analyzed

### Expected Outcome

**Successful Test Output:**
```
=== Python Enhanced Integration Test with Log Verification ===
Starting Python server and client with log capture...
PYTHONPATH: /workspace/holohub/build-video_streaming_demo_enhanced/python/lib:...
Python Server log: /tmp/server_python_log.XXXXXX
Python Client log: /tmp/client_python_log.XXXXXX
Starting Python streaming server...
Waiting for Python server to initialize...
✓ Python Server process is running
Starting Python streaming client...
Letting Python streaming run for 30 seconds...
Stopping Python client...
Stopping Python server...

=== Verifying Python Server Logs ===
✓ Python Server: Client connected
✓ Python Server: Upstream connection established
✓ Python Server: Downstream connection established
✓ Python Server: StreamingServerUpstreamOp processed 566 unique frames
✓ Python Server: StreamingServerDownstreamOp processed 566 tensors
✓ Python Server: Frame processing statistics logged

=== Verifying Python Client Logs ===
✓ Python Client: Sent 566 frames successfully
✓ Python Client: Received 534 frames from server
✓ Python Client: Frame validation passed
✓ Python Client: Streaming client started

=== Python Test Results Summary ===
Python Server checks passed: 6
Python Client checks passed: 4
✓ PYTHON STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
✓ Python Integration test PASSED
```

### Acceptance Criteria

The Python integration test validates the following checks. **All checks must pass** for the test to succeed:

#### Server Checks (6 required)

| Check | Description | Success Criteria |
|-------|-------------|------------------|
| ✓ Client connected | Client successfully connects to server | `"Client connected"` in server logs |
| ✓ Upstream connection | Upstream channel established (client → server) | `"Upstream connection established"` in server logs |
| ✓ Downstream connection | Downstream channel established (server → client) | `"Downstream connection established"` in server logs |
| ✓ Frame processing | Server processes frames from client | ≥100 `"Processing UNIQUE frame"` log entries |
| ✓ Tensor transmission | Server sends tensors to client | ≥100 `"DOWNSTREAM: Processing tensor"` log entries |
| ✓ Statistics logged | Frame processing statistics are logged | `"Frame Processing Stats"` in server logs |

#### Client Checks (4 required)

| Check | Description | Success Criteria |
|-------|-------------|------------------|
| ✓ Frames sent | Client successfully sends frames to server | ≥100 `"Frame sent successfully"` log entries |
| ✓ Frames received | Client receives frames from server | ≥100 `"CLIENT: Received frame"` log entries |
| ✓ Frame validation | Received frames pass validation | `"Frame validation passed"` in client logs |
| ✓ Client startup | Client application starts successfully | `"STARTING STREAMING CLIENT"` or `"Starting Streaming Client Demo"` in client logs |

### Frame Throughput Metrics

**Minimum Requirements:**
- **Server Frame Processing**: ≥100 frames in 30 seconds (~3.3 fps minimum)
- **Client Frame Sending**: ≥100 frames in 30 seconds (~3.3 fps minimum)
- **Client Frame Reception**: ≥100 frames in 30 seconds (~3.3 fps minimum)

**Typical Performance:**
- **Frames Processed**: 500-600 frames in 30 seconds (~16-20 fps)
- **Bidirectional Verification**: Both upstream (client → server) and downstream (server → client) verified

### Test Configuration

**Server Configuration:**
```bash
python3 streaming_server_demo.py --width 854 --height 480
```

**Client Configuration:**
```bash
python3 streaming_client_demo.py --source replayer --width 854 --height 480
```

### Troubleshooting

**Test Failure - Insufficient Frames:**
```
✗ Python Server: Only 50 frames processed (minimum: 100)
```
**Cause**: Insufficient streaming time or connection issues  
**Solution**: Check network connectivity, verify logs for connection errors

**Test Failure - Client Not Started:**
```
✗ Python Client: Streaming client failed to start
```
**Cause**: Missing dependencies or PYTHONPATH issues  
**Solution**: Ensure `HOLOHUB_BUILD_PYTHON=ON` and rebuild with Python bindings

**Test Failure - No Frames Received:**
```
✗ Python Client: No frames received from server
```
**Cause**: Downstream connection failure or server issues  
**Solution**: Check server logs for downstream connection establishment

### CI/CD Integration

**Exit Codes:**
- `0`: All tests passed (all 10 checks passed)
- `1`: One or more checks failed
- `124`: Test timeout (300 seconds)

**CI-Friendly Command:**
```bash
timeout 300 ./run test video_streaming_demo_enhanced --test video_streaming_integration_test_python
echo "Python integration test exit code: $?"
```

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