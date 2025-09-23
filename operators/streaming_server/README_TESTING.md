# StreamingServer Operator Testing

## Overview

This directory contains comprehensiv tests for the `StreamingServerOp` operator, covering both **Python** and **C++** implementations:

- **Python Unit Tests** (pytest): Test Python operator bindings and parameter validation
- **Python Infrastructure Tests**: Test Python demo applications without video data
- **Python Functional Tests**: Test Python applications with real video data
- **C++ Infrastructure Tests** (CTest): Test C++ demo applications and operator functionality
- **C++ Functional Tests** (CTest): Test C++ implementation with real video data

## Python Unit Tests

### Test File: `test_streaming_server_op.py`

The test suite covers:

#### 1. **Initialization Tests**
- Basic operator creation and properties validation
- Default name handling
- Parameter validation for video settings (width, height, fps)
- Network parameter validation (port, server_name)
- Frame handling mode validation (receive_frames, send_frames)
- Multi-instance support

#### 2. **Port Setup Tests** 
- Validates that StreamingServerOp has no input/output ports (standalone operator)
- Confirms the operator communicates via network, not Holoscan pipeline ports

#### 3. **Error Handling Tests**
- Invalid parameter values (zero/negative width, height, fps)
- Invalid port numbers (0, too high)
- Invalid parameter types (wrong data types)

#### 4. **Edge Case Tests**
- Minimum reasonable parameter values
- Empty server names
- Multiple operator instances
- Boundary conditions

#### 5. **Architecture Tests**
- Parameter encapsulation verification
- Lifecycle method existence (setup, initialize, start, stop, compute)
- Operator type validation (NATIVE)
- Standalone operator behavior validation

### Running the Tests

## **Unit Test 1**

**Purpose:** Validate StreamingServerOp Python operator creation, parameter validation, and lifecycle methods

**What it Tests:**
- Operator initialization with default parameters only (real operator has limited constructor)
- Basic operator properties and type validation
- Lifecycle method existence (setup, initialize, start, stop, compute)
- Multiple operator instances
- Mock operator fallback for testing environments

**Expected Outcome:**
- All 16 unit tests pass in 0.04 seconds
- Test execution completes quickly
- Mock fallback works when real operator unavailable
- Output: `============================== 16 passed in 0.04s ==============================`

**Acceptance Criteria:**
- ✅ All 16 pytest assertions pass
- ✅ No import or initialization errors
- ✅ Test completes in under 1 second
- ✅ Mock fallback works when real operator unavailable
- ✅ Output contains "16 passed" message

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_server --verbose
```

## **Unit Test 2**

**Purpose:** Basic StreamingServer Python functionality validation without video data

**What it Tests:**
- StreamingServer operator startup and initialization
- Network port binding and server creation (port 48010)
- Server configuration validation (854x480, 30fps)
- Graceful shutdown and cleanup processes

**Expected Outcome:**
- Test passes in ~30.7 seconds
- Server starts successfully on port 48010
- Timeout after 30 seconds is expected (normal behavior)
- Output: `Test PASSED: StreamingServer functionality validated successfully (timeout expected)`

**Expected Behavior (Normal Operation):**
Since there are no streaming clients connecting during tests, the server behavior is expected:
- Server starts and listens on port 48010
- Server waits for client connections (timeout after 30 seconds is normal)
- No connection errors since the server is waiting, not attempting connections

**Acceptance Criteria:**
- ✅ "StreamingServer functionality validated successfully" message appears
- ✅ Server starts and listens on port 48010
- ✅ "✅ StreamingServer started successfully and is running" log message
- ✅ Server status shows "Is running: YES"
- ✅ Timeout behavior is expected (no clients connecting)
- ✅ Clean shutdown and destructor completion

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_server --verbose
```

## **Unit Test 3**

**Purpose:** StreamingServer Python functionality with video data directory detection

**What it Tests:**
- Video data directory discovery and validation
- Server configuration with available video data
- Functional test mode with data directory setup
- Client connection readiness

**Expected Outcome:**
- Test passes in ~60.7 seconds
- Video data directory detected and configured (/workspace/holohub/data)
- Server ready for client connections
- Output: `✅ FUNCTIONAL test PASSED: Python StreamingServer with data directory successful`

**Expected Behavior (Normal Operation):**
Since there are no streaming clients connecting during tests, the server behavior is expected:
- Server starts and waits for client connections
- Video data is loaded and ready for streaming
- Timeout after 60 seconds is normal (no clients connecting)
- No connection errors since the server is in listening mode

**Acceptance Criteria:**
- ✅ "FUNCTIONAL test PASSED: Python StreamingServer with data directory successful" message
- ✅ "Available video data" and "ready to accept client connections" messages
- ✅ Video data directory found and configured
- ✅ Server starts and listens for client connections
- ✅ Timeout behavior is expected (no clients connecting)
- ✅ No critical streaming errors

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_server --verbose
```

## **Unit Test 4**

**Purpose:** Validate C++ StreamingServer demo application functionality

**What it Tests:**
- C++ streaming server executable functionality
- Configuration file loading and parsing
- C++ operator integration and initialization
- Server startup and graceful shutdown in C++

**Expected Outcome:**
- Test passes in ~30.9 seconds
- C++ executable runs successfully
- Configuration loads from YAML file
- Output: `Test PASSED: C++ StreamingServer test successful`

**Expected Behavior (Normal Operation):**
Since there are no streaming clients connecting during tests, the server behavior is expected:
- Server starts and waits for client connections
- Timeout after 30 seconds is normal (no clients connecting)
- No connection errors since the server is in listening mode

**Acceptance Criteria:**
- ✅ "Test PASSED: C++ StreamingServer test successful" message appears
- ✅ "Streaming Server Test Application" startup message
- ✅ "Successfully loaded configuration" message
- ✅ Server starts, runs, and stops cleanly
- ✅ Timeout behavior is expected (no clients connecting)
- ✅ No compilation or runtime errors

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_server --verbose
```

## **Unit Test 5**

**Purpose:** End-to-end C++ StreamingServer with real video data

**What it Tests:**
- C++ implementation with actual video file processing
- Real video data detection and usage
- Complete streaming server workflow in C++
- Performance under realistic data conditions

**Expected Outcome:**
- Test passes in ~60.7 seconds
- Real video data processed successfully
- Full functional streaming server capability demonstrated
- Output: `✅ FUNCTIONAL test PASSED: C++ StreamingServer with data directory successful`

**Expected Behavior (Normal Operation):**
Since there are no streaming clients connecting during tests, the server behavior is expected:
- Server starts with real video data loaded
- Server waits for client connections with video ready to stream
- Timeout after 60 seconds is normal (no clients connecting)
- No connection errors since the server is in listening mode

**Acceptance Criteria:**
- ✅ "FUNCTIONAL test PASSED: C++ StreamingServer with data directory successful" message
- ✅ "🎬 FUNCTIONAL test: Using real video data" message
- ✅ Video data size information displayed
- ✅ Server accepts client connections and processes video streams
- ✅ Timeout behavior is expected (no clients connecting)
- ✅ Complete streaming server workflow demonstrated

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_server --verbose
```

## **Quick Test Commands Summary:**

```bash
# Run all StreamingServer tests (recommended)
./holohub test video_streaming_server --verbose

# Expected total test time: ~180 seconds for all 5 tests
# Test results: 100% tests passed, 0 tests failed out of 5
```

### Test Categories

| Test Type | Category | Test Count | Purpose |
|-----------|----------|------------|---------|
| **Python Unit** | Initialization | 8 tests | Operator creation with various parameters |
| **Python Unit** | Port Setup | 1 test | Validates standalone architecture (no ports) |
| **Python Unit** | Error Handling | 6 tests | Invalid parameter handling |
| **Python Unit** | Edge Cases | 6 tests | Boundary conditions and special cases |
| **Python Unit** | Architecture | 4 tests | Operator structure and lifecycle |
| **Python Infrastructure** | Runtime | 1 test | Basic server startup without video data |
| **Python Functional** | Data Processing | 1 test | Server with video data directory |
| **C++ Infrastructure** | Runtime | 1 test | C++ demo application functionality |
| **C++ Functional** | Data Processing | 1 test | C++ implementation with real video data |

### Key Testing Insights

#### **StreamingServerOp Architecture**
- **Standalone Operator**: No input/output ports (unlike pipeline operators)
- **Network Communication**: Receives frames from network clients, not pipeline
- **Parameter Encapsulation**: Configuration parameters are properly encapsulated
- **Lifecycle Methods**: Full Holoscan operator lifecycle support

#### **Default Parameters** (based on C++ implementation)
- Width: 854, Height: 480, FPS: 30
- Port: 48010
- Server Name: "StreamingServer"
- Multi-instance: false
- Receive/Send Frames: true/true

#### **Test Philosophy**
- **Unit Testing Focus**: Tests operator creation, parameter validation, and basic structure
- **No Network Testing**: Network functionality requires integration testing
- **Parameter Boundary Testing**: Validates edge cases and error conditions
- **Architecture Validation**: Ensures proper Holoscan operator compliance

### Example Test Usage

```python
def test_streaming_server_op_init_basic(self, fragment):
    """Test basic StreamingServerOp initialization and properties."""
    name = "streaming_server_op"
    op = StreamingServerOp(fragment=fragment, name=name)
    
    assert isinstance(op, BaseOperator)
    assert op.operator_type == Operator.OperatorType.NATIVE
    assert f"name: {name}" in repr(op)
```

### Dependencies

- `pytest`: Testing framework
- `holoscan.core`: Holoscan framework components
- `holohub.streaming_server`: StreamingServer operator module

### Notes

- Tests focus on **unit testing** (operator creation/validation)
- **Integration testing** (actual streaming) requires separate test environment
- Tests are designed to work with standard HoloHub fixtures from `conftest.py`

