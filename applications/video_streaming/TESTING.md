# Video Streaming Integration Testing

This document provides comprehensive documentation for integration testing of the video streaming application, including both C++ and Python implementations.

## Overview

The video streaming demo includes integration testing to verify end-to-end functionality between client and server components.

The integration test validates:

- **Server Startup**: Streaming server initializes and starts listening
- **Client Connection**: Streaming client connects to server successfully  
- **Video Streaming**: Bidirectional video frame transmission (client→server→client)
- **Resource Management**: Proper cleanup and resource handling
- **Error Handling**: Graceful handling of connection issues

## Running Integration Tests

### Option 1: Using Integration Test Script

The integration test script (`integration_test.sh`) runs the complete end-to-end test in a Docker container with proper SDK version and dependencies.

```bash
./applications/video_streaming/integration_test.sh
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

### Option 2: Using HoloHub CLI

```bash
# From holohub root - standard HoloHub test command
./holohub test video_streaming \
  --ctest-options="-R video_streaming_integration_test"
```

**Note:** Both methods run the same underlying integration test defined in `CMakeLists.txt`. The wrapper script (`integration_test.sh`) adds developer-friendly conveniences on top of the direct command.

## Integration Test Process

The integration test (whether run via wrapper script or direct command) follows this sequence:

### 1. Pre-Test Setup (10-20 seconds)

```bash
# Displays current git commit
echo "Current commit: $(git log --oneline -1)"

# Cleans Docker build cache (optional, for fresh builds)
docker system prune -f --filter "label=holohub"

# Sets SDK version environment variable
export HOLOHUB_BASE_SDK_VERSION=3.5.0
```

### 2. Docker Build & Test Execution (2-4 minutes)

```bash
# Builds Docker image and runs CTest
./holohub test video_streaming \
  --cmake-options="-DBUILD_TESTING=ON" \
  --ctest-options="-R video_streaming_integration_test -V" \
  --verbose
```

**What happens internally:**

- Builds Docker image with Holoscan SDK 3.5.0
- Compiles server and client C++ applications
- Copies configuration files to build directory
- Runs CTest with the integration test

### 3. Integration Test Execution (44 seconds)

The `video_streaming_integration_test` defined in CMakeLists.txt:

1. **Server Startup** (10 seconds)
   - Launches streaming server in background: `streaming_server_demo`
   - Uses config: `streaming_server_demo.yaml`
   - Waits for server to initialize and stabilize

2. **Client Connection & Streaming** (30 seconds)
   - Starts streaming client: `streaming_client_demo`
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

### 4. Post-Test Analysis (5 seconds)

```bash
# Verifies test results from log file
if grep -q "Test.*Passed\|100% tests passed, 0 tests failed" integration_test.log; then
  echo "✓ Integration test PASSED"
  exit 0
fi
```

## Success Criteria

The integration test **PASSES** when **ALL 10 checks** are met (6 server + 4 client):

### ✅ Server Log Verification Criteria (6 checks required)

1. **Client Connected**: `grep -q 'Client connected' $SERVER_LOG`
   - Verifies a client successfully connected to the server

2. **Upstream Connection Established**: `grep -q 'Upstream connection established' $SERVER_LOG`
   - Verifies the upstream data channel (client→server) is established

3. **Downstream Connection Established**: `grep -q 'Downstream connection established' $SERVER_LOG`
   - Verifies the downstream data channel (server→client) is established

4. **Upstream Frame Processing**: `grep -q 'Processing UNIQUE frame' $SERVER_LOG`
   - Verifies StreamingServerUpstreamOp received and processed frames from client
   - Typical: 565-567 unique frames in 30 seconds

5. **Downstream Tensor Processing**: `grep -q 'DOWNSTREAM: Processing tensor' $SERVER_LOG`
   - Verifies StreamingServerDownstreamOp processed and sent tensors to client
   - Typical: 565-567 tensors in 30 seconds

6. **Frame Processing Statistics**: `grep -q 'Frame Processing Stats' $SERVER_LOG`
   - Verifies the server logged performance statistics at shutdown

### ✅ Client Log Verification Criteria (4 checks required)

1. **Frame Sending Success**: `grep -q 'Frame sent successfully' $CLIENT_LOG`
   - Verifies client successfully sent frames to server
   - Typical: 565-567 frames sent in 30 seconds

2. **Frame Reception Success**: `grep -q 'CLIENT: Received frame' $CLIENT_LOG`
   - Verifies client successfully received frames from server (bidirectional)
   - Typical: 533-540 frames received in 30 seconds (slight lag expected)
   - Completes end-to-end bidirectional verification

3. **Frame Validation**: `grep -q 'Frame validation passed' $CLIENT_LOG`
   - Verifies client frame validation logic is working
   - Logs appear every 30 frames (~1 second at 30 FPS)

4. **Streaming Client Started**: `grep -q 'STARTING STREAMING CLIENT' $CLIENT_LOG`
   - Verifies client initialization completed successfully

### ✅ Overall Test Success

**Test PASSES if:**

- Server checks: **6/6 passed** (required: 6)
- Client checks: **4/4 passed** (required: 4)
- Total: **10/10 checks passed**

**Test FAILS if:**

- Any check fails (server < 6 or client < 4)
- Test times out (> 300 seconds)

**Note on Segmentation Faults:**

- Segmentation faults may appear during graceful shutdown (after SIGTERM)
- These are expected and do NOT cause test failure
- The test passes if all 10 log verification checks succeed
- Example: `Segmentation fault (core dumped)` appears in lines 423 and 680 of test output

## Expected Output

### Console Output (Successful Test)

```bash
=== Video Streaming Demo Integration Test ===
This test may take up to 10 minutes to complete...
NOTE: Test runs in Docker and uses committed source code (not local build)

[Docker build output...]
Step 1/15 : ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu
[...]

[CTest output...]
Test project /workspace/holohub/build-video_streaming
    Start 1: video_streaming_integration_test

1: === Integration Test with Log Verification ===
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
1: /usr/bin/bash: line 53:   855 Segmentation fault      (core dumped) [...]
1: 
1: === Verifying Server Logs ===
1: ✓ Server: Client connected
1: ✓ Server: Upstream connection established
1: ✓ Server: Downstream connection established
1: ✓ Server: StreamingServerUpstreamOp processed 567 unique frames
1: ✓ Server: StreamingServerDownstreamOp processed 567 tensors
1: ✓ Server: Frame processing statistics logged
1: 
1: === Verifying Client Logs ===
1: ✓ Client: Sent 567 frames successfully
1: ✓ Client: Received 535 frames from server
1: ✓ Client: Frame validation passed
1: ✓ Client: Streaming client started
1: 
1: === Test Results Summary ===
1: Server checks passed: 6
1: Client checks passed: 4
1: ✓ STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
1: ✓ Integration test PASSED
1/2 Test #1: video_streaming_integration_test ..........   Passed   44.08 sec

    Start 2: video_streaming_integration_test_python

2: === Python Integration Test with Log Verification ===
2: Starting Python server and client with log capture...
2: PYTHONPATH: /workspace/holohub/build-video_streaming/python/lib:...
2: Python Server log: /tmp/server_python_log.XXXXXX
2: Python Client log: /tmp/client_python_log.XXXXXX
2: Starting Python streaming server...
2: Waiting for Python server to initialize...
2: ✓ Python Server process is running
2: Starting Python streaming client...
2: Letting Python streaming run for 30 seconds...
2: Stopping Python client...
2: Stopping Python server...
2: /usr/bin/bash: line 58:  1144 Segmentation fault      (core dumped) [...]
2: 
2: === Verifying Python Server Logs ===
2: ✓ Python Server: Client connected
2: ✓ Python Server: Upstream connection established
2: ✓ Python Server: Downstream connection established
2: ✓ Python Server: StreamingServerUpstreamOp processed 565 unique frames
2: ✓ Python Server: StreamingServerDownstreamOp processed 565 tensors
2: ✓ Python Server: Frame processing statistics logged
2: 
2: === Verifying Python Client Logs ===
2: ✓ Python Client: Sent 565 frames successfully
2: ✓ Python Client: Received 533 frames from server
2: ✓ Python Client: Frame validation passed
2: ✓ Python Client: Streaming client started
2: 
2: === Python Test Results Summary ===
2: Python Server checks passed: 6
2: Python Client checks passed: 4
2: ✓ PYTHON STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
2: ✓ Python Integration test PASSED
2/2 Test #2: video_streaming_integration_test_python ...   Passed   44.39 sec

The following tests passed:
    video_streaming_integration_test
    video_streaming_integration_test_python

100% tests passed, 0 tests failed out of 2

Total Test time (real) =  88.48 sec

=== VERIFICATION ===
✓ Integration test passed with detailed verification
✓ Server component verified
✓ Client component verified
✓ Integration test PASSED
```

### Key Log Patterns to Look For

**Server Success Indicators:**

```console
[info] ✅ [UPSTREAM 12345] Client connected: connection details
[info] ⬆️ [UPSTREAM 12346] Upstream connection established: connection details
[info] ⬇️ [DOWNSTREAM 12347] Downstream connection established: connection details
[info] ✅ Processing UNIQUE frame: 854x480, 1639680 bytes, timestamp=29938
[info] 📊 DOWNSTREAM: Processing tensor 567 - shape: 480x854x4, 1639680 bytes
[info] ✅ DOWNSTREAM: Frame sent successfully to StreamingServerResource
[info] 📊 Frame Processing Stats: Total=567, Unique=567, Duplicates=0
```

**Client Success Indicators:**

```console
[info] Source set to: replayer
[info] Using video replayer as source
[info] 🔧 ENHANCED StreamingClient constructed! Version with buffer validation fixes!
[info] StreamingClient created successfully
[info] ✅ Connection established successfully
[info] ✅ Upstream connection established successfully!
[info] ✅ Frame sent successfully on attempt 1
[info] 🎯 CLIENT: Frame received callback triggered! Frame: 854x480, 1639680 bytes
[info] 📥 CLIENT: Received frame #533 from server: 854x480
```

**Performance Indicators:**

```console
# Server processed 565-567 frames in both directions
[info] ✅ Processing UNIQUE frame: 854x480, 1639680 bytes, timestamp=29938
[info] 📊 DOWNSTREAM: Processing tensor 567 - shape: 480x854x4, 1639680 bytes

# Client sent 565-567 frames and received ~533 frames
[info] ✅ Frame sent successfully on attempt 1
[info] 📥 CLIENT: Received frame #533 from server: 854x480

# Frame rate: ~19 FPS (567 frames ÷ 30 seconds)
# Bidirectional throughput: ~62 MB/s (1.64MB per frame × 19 FPS × 2 directions)
```

### Integration Test Log File

The complete test execution is saved to `integration_test.log` (typically 700-800 lines). This file contains:

1. **Docker Build Logs**: Complete build output with all dependencies (~200 lines)
2. **CMake Configuration**: Build configuration and test setup (~100 lines)
3. **CTest Execution**: Detailed test execution with timestamps (~400 lines)
4. **Test Verification**: Log verification checks with pass/fail status (~100 lines)
5. **Test Summary**: Final PASS/FAIL status with verification details

**Note**: The actual server and client application logs are redirected to temporary files during testing and are NOT included in `integration_test.log`. These detailed logs are only displayed if the test fails.

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

## Troubleshooting Integration Tests

### Common Issues and Solutions

**Test Failure: Connection Events Not Logged**

If you see output like:

```bash
=== Verifying Server Logs ===
✗ Server: Upstream connection not established
✗ Server: Downstream connection not established
✓ Server: StreamingServerUpstreamOp processed 567 unique frames  # But frames work!
✓ Server: StreamingServerDownstreamOp processed 567 tensors      # But frames work!

=== Verifying Client Logs ===
✓ Client: Sent 567 frames successfully
✓ Client: Received 535 frames from server  # Bidirectional works!

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
./holohub build video_streaming --language cpp
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
1: Segmentation fault (core dumped) ./streaming_server_demo
1: ✓ Server: StreamingServerUpstreamOp processed 567 unique frames
1: ✓ STREAMING VERIFICATION PASSED - Frames actually transmitted!
```

- Segfault occurs during cleanup after test completes
- Test passes if all 10 checks passed before shutdown
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

This file contains all information needed for debugging failed tests (~700-800 lines for a complete run).

### Continuous Integration

The integration test is designed for CI/CD pipelines:

```bash
# CI-friendly command with timeout and exit codes
timeout 300 ./applications/video_streaming/integration_test.sh
echo "Integration test exit code: $?"
```

**Exit Codes:**
- `0`: All tests passed successfully
- `1`: Test failures detected
- `124`: Test timeout (5 minutes)

---

## Python Integration Testing

### Overview

The Python integration test validates the complete bidirectional video streaming pipeline using Python implementations of both the server and client applications. The test verifies frame transmission, reception, and processing statistics.

### Running the Python Integration Test

**Command:**
```bash
# From holohub root - run Python integration test
./holohub test video_streaming \
  --docker-file applications/video_streaming/Dockerfile \
  --cmake-options='-DHOLOHUB_BUILD_PYTHON=ON -DBUILD_TESTING=ON' \
  --ctest-options="-R video_streaming_integration_test_python -VV"
```

**Test Duration:** ~44 seconds (30 seconds of streaming + setup/teardown)

**Requirements:**
- Docker and NVIDIA GPU
- Testing enabled via `--cmake-options='-DBUILD_TESTING=ON'`
- Python bindings enabled via `--cmake-options='-DHOLOHUB_BUILD_PYTHON=ON'`
- Custom Dockerfile for OpenSSL 3.4.0 dependencies

### Test Workflow

1. **Server Startup**: Python streaming server starts with 854x480 resolution
2. **Client Startup**: Python streaming client starts with replayer source (854x480)
3. **Streaming Duration**: 30 seconds of bidirectional video streaming
4. **Log Verification**: Comprehensive log analysis for both server and client
5. **Graceful Shutdown**: Both processes are stopped and logs are analyzed

### Expected Outcome

**Successful Test Output (Standalone Python Test):**
```
=== Python Integration Test with Log Verification ===
Starting Python server and client with log capture...
PYTHONPATH: /workspace/holohub/build-video_streaming/python/lib:...
Python Server log: /tmp/server_python_log.XXXXXX
Python Client log: /tmp/client_python_log.XXXXXX
Starting Python streaming server...
Waiting for Python server to initialize...
✓ Python Server process is running
Starting Python streaming client...
Letting Python streaming run for 30 seconds...
Stopping Python client...
Stopping Python server...
/usr/bin/bash: line 58:  530 Segmentation fault      (core dumped) python3 streaming_server_demo.py ...

=== Verifying Python Server Logs ===
✓ Python Server: Client connected
✓ Python Server: Upstream connection established
✓ Python Server: Downstream connection established
✓ Python Server: StreamingServerUpstreamOp processed 564 unique frames
✓ Python Server: StreamingServerDownstreamOp processed 564 tensors
✓ Python Server: Frame processing statistics logged

=== Verifying Python Client Logs ===
✓ Python Client: Sent 564 frames successfully
✓ Python Client: Received 531 frames from server
✓ Python Client: Frame validation passed
✓ Python Client: Streaming client started

=== Python Test Results Summary ===
Python Server checks passed: 6
Python Client checks passed: 4
✓ PYTHON STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
✓ Python Integration test PASSED

1/1 Test #2: video_streaming_integration_test_python ...   Passed   44.41 sec

The following tests passed:
        video_streaming_integration_test_python

100% tests passed, 0 tests failed out of 1

Total Test time (real) = 44.42 sec
```

**Note:** When running the full integration test suite (both C++ and Python tests together), you'll see:
- Test counter: `2/2 Test #2` (instead of `1/1`)
- Both tests listed: `video_streaming_integration_test` and `video_streaming_integration_test_python`
- Total: `0 tests failed out of 2`
- Total time: ~88 seconds (both tests combined)

**Important Notes:**
- A segmentation fault may occur during shutdown - this is **expected** and does not indicate test failure
- The test explicitly ignores cleanup segfaults using `wait $PID || true`
- Test result is based **solely** on the 10 verification checks, not process exit codes
- CTest will show **"Passed"** if all checks succeed, even with segfault during cleanup

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

#### Overall Test Success

**Test PASSES when:**
- ✅ All 6 server checks pass (6/6)
- ✅ All 4 client checks pass (4/4)
- ✅ Total: **10/10 checks passed**
- ✅ Exit code: `0`
- ✅ CTest output: `"100% tests passed, 0 tests failed out of 1"`

**Test FAILS when:**
- ❌ Any server check fails (< 6 passed)
- ❌ Any client check fails (< 4 passed)
- ❌ Total: < 10 checks passed
- ❌ Exit code: `1`
- ❌ CTest output: `"Tests failed with return value: -1"`

**Note on Segfaults:** Segmentation faults during cleanup do **not** cause test failure. The test uses `wait $PID || true` to ignore non-zero exit codes from cleanup. Only the 10 verification checks determine pass/fail status.

### Frame Throughput Metrics

**Minimum Requirements:**
- **Server Frame Processing**: ≥100 frames in 30 seconds (~3.3 fps minimum)
- **Client Frame Sending**: ≥100 frames in 30 seconds (~3.3 fps minimum)
- **Client Frame Reception**: ≥100 frames in 30 seconds (~3.3 fps minimum)

**Typical Performance:**
- **Frames Processed**: 500-600 frames in 30 seconds (~16-20 fps)
- **Bidirectional Verification**: Both upstream (client → server) and downstream (server → client) verified


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

**Segmentation Fault at Shutdown (Expected Behavior):**
```bash
Segmentation fault (core dumped) python3 streaming_server_demo.py
✓ Python Server: StreamingServerUpstreamOp processed 561 unique frames
✓ PYTHON STREAMING VERIFICATION PASSED - All checks passed, frames transmitted!
✓ Python Integration test PASSED
```

**Explanation**: A segfault occurs during Python interpreter shutdown when cleaning up C++ resources (StreamingServerResource, CUDA contexts). This is a known issue with destruction order in Python bindings and does **not affect streaming functionality**.

- Segfault happens **after** all streaming completes successfully
- Test uses `wait $PID || true` to ignore non-zero exit codes during cleanup
- Test passes if all 10 verification checks pass before shutdown
- **Solution**: No action needed - this is expected behavior and the test correctly reports PASS

### CI/CD Integration

**Exit Codes:**
- `0`: All tests passed (all 10 checks passed)
- `1`: One or more checks failed
- `124`: Test timeout (300 seconds)

**CI-Friendly Command:**
```bash
timeout 300 ./holohub test video_streaming \
  --docker-file applications/video_streaming/Dockerfile \
  --cmake-options='-DHOLOHUB_BUILD_PYTHON=ON -DBUILD_TESTING=ON' \
  --ctest-options="-R video_streaming_integration_test_python"
echo "Python integration test exit code: $?"
```

## See Also

- **[Main README](README.md)** - Application overview and usage
- **[Client README](video_streaming_client/README.md)** - Client-specific documentation
- **[Server README](video_streaming_server/README.md)** - Server-specific documentation

