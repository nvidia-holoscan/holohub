# StreamingClient Operator Testing

## Overview

This directory contains comprehensive tests for the `StreamingClientOp` operator, covering both **Python** and **C++** implementations:

- **Python Unit Tests** (pytest): Test Python operator bindings and parameter validation
- **Python Infrastructure Tests**: Test Python demo applications without video data
- **Python Functional Tests**: Test Python applications with real video data
- **C++ Infrastructure Tests** (CTest): Test C++ demo applications and operator functionality

## Python Unit Tests

### Test File: `test_streaming_client_op.py`

The test suite covers:

#### 1. **Initialization Tests**
- Basic operator creation and properties validation
- Default name handling
- Parameter validation for video settings (width, height, fps)
- Network parameter validation (server_ip, signaling_port)
- Frame handling mode validation (receive_frames, send_frames)

#### 2. **Port Setup Tests** 
- Validates that StreamingClientOp has input/output ports (pipeline operator)
- Confirms the operator processes video frames through Holoscan pipeline ports

#### 3. **Error Handling Tests**
- Invalid parameter values (zero/negative width, height, fps)
- Invalid port numbers (0, too high)
- Invalid parameter types (wrong data types)

#### 4. **Edge Case Tests**
- Minimum reasonable parameter values
- Empty server IP addresses
- Multiple operator instances
- Boundary conditions

#### 5. **Architecture Tests**
- Parameter encapsulation verification
- Lifecycle method existence (setup, initialize, start, stop, compute)
- Operator type validation (NATIVE)
- Pipeline operator behavior validation

## Integration Tests

### Test Directory: `tests/`

Contains functional and integration tests for complete streaming workflows:

#### **Python Tests**
- `run_test.sh` - Basic infrastructure test for StreamingClient functionality
- `run_functional_test.sh` - Full functional test with real video data
- `video_streaming_client_functional.py` - Python functional test application

#### **C++ Tests**
- `run_cpp_test.sh` - C++ infrastructure test for streaming client demo

#### **Test Assets**
- `testing/` directory contains:
  - Golden reference frames (`0001.png` - `0010.png`)
  - `generate_golden_frames.py` - Script to generate reference frames
  - `video_streaming_client_testing.yaml` - Test configuration

### Running the Tests

## **Test 1: Python Unit Tests**

**Purpose:** Validate StreamingClientOp Python operator creation, parameter validation, and lifecycle methods

**What it Tests:**
- Operator initialization with various parameters (video resolution, network settings, frame modes)
- Parameter encapsulation and validation
- Lifecycle method existence (setup, initialize, start, stop, compute)
- Multiple operator instances
- Edge cases and error handling

**How to build and run the test:**
```bash
./holohub test video_streaming_client --verbose
```

**Expected Outcome:** 
- All 28+ unit tests pass
- Test execution completes in < 1 second
- No import or initialization errors

**Acceptance Criteria:**
- ✅ All pytest assertions pass
- ✅ Mock fallback works when real operator unavailable
- ✅ Parameter validation catches invalid inputs
- ✅ Multiple operator instances can be created

## **Test 2: Python Infrastructure Tests**

**Purpose:** Basic Python functionality validation without video data

**What it Tests:**
- StreamingClient operator can start and initialize properly
- Network configuration and connection setup
- Graceful shutdown and cleanup
- Basic pipeline functionality

**How to build and run the test:**
```bash
./holohub test video_streaming_client --verbose
```

**Expected Outcome:**
- Test passes within 30-120 seconds
- StreamingClient initializes successfully
- No critical errors during startup/shutdown

**Acceptance Criteria:**
- ✅ "Test PASSED: StreamingClient functionality validated" message appears
- ✅ No fatal errors or exceptions
- ✅ Clean shutdown without hanging

## **Test 3: Python Functional Tests**

**Purpose:** End-to-end Python streaming functionality with real video data

**What it Tests:**
- StreamingClient processing with actual video frames
- Client-server communication protocols
- Video data handling and frame processing
- Performance under realistic conditions

**How to build and run the test:**
```bash
./holohub test video_streaming_client --verbose
```

**Expected Outcome:**
- Test passes within 60-120 seconds
- Video frames processed successfully
- Client connects to server and streams data

**Acceptance Criteria:**
- ✅ "FUNCTIONAL test.*successful" regex match
- ✅ Video data directory found and used
- ✅ No streaming protocol errors

## **Test 4: C++ Infrastructure Tests**

**Purpose:** Validate C++ StreamingClient demo application functionality

**What it Tests:**
- C++ implementation of streaming client
- Binary executable functionality
- Configuration file loading
- C++ operator integration

**How to build and run the test:**
```bash
./holohub test video_streaming_client --verbose
```

**Expected Outcome:**
- C++ executable runs successfully
- Configuration loads properly
- Test completes within 90 seconds

**Acceptance Criteria:**
- ✅ "Test PASSED: C\\+\\+ StreamingClient.*test successful" regex match
- ✅ No compilation or runtime errors
- ✅ Clean application shutdown

## **Quick Test Commands Summary:**

```bash
# Run all StreamingClient tests
./holohub test video_streaming_client --verbose
```

### Test Categories

| Test Type | Category | Test Count | Purpose |
|-----------|----------|------------|---------|
| **Python Unit** | Initialization | 15 tests | Operator creation with various parameters |
| **Python Unit** | Port Setup | 1 test | Input port validation (video input) |
| **Python Unit** | Error Handling | 4 tests | Invalid parameter handling |
| **Python Unit** | Edge Cases | 6 tests | Boundary conditions and special cases |
| **Python Unit** | Architecture | 6+ tests | Operator structure and lifecycle |
| **Python Infrastructure** | Runtime | 1 test | Basic client startup without video data |
| **Python Functional** | Data Processing | 1 test | Client with video data processing |
| **C++ Infrastructure** | Runtime | 1 test | C++ demo application functionality |

### Key Testing Insights

#### **StreamingClientOp Architecture**
- **Pipeline Operator**: Has input/output ports for video frame processing
- **Bidirectional Communication**: Can send frames to and receive frames from servers
- **Parameter Encapsulation**: Configuration parameters are properly encapsulated
- **Lifecycle Methods**: Full Holoscan operator lifecycle support

#### **Default Parameters** (based on C++ implementation)
- Width: 854, Height: 480, FPS: 30
- Server IP: "localhost"
- Signaling Port: 48010
- Receive/Send Frames: true/true

#### **Test Modes**
1. **Infrastructure Mode**: Tests operator functionality without video data
2. **Functional Mode**: Tests complete video processing pipeline
3. **Unit Mode**: Tests operator creation and parameter validation

### Test Infrastructure

#### **Video Data Requirements**
- Tests can run in infrastructure mode without video data
- Functional tests require `surgical_video.gxf_index` and `surgical_video.gxf_entities`
- Falls back gracefully when video data is not available

#### **Golden Frame Testing**
- Reference frames stored in `testing/` directory
- `generate_golden_frames.py` creates new reference frames
- Visual regression testing for frame processing validation

#### **Mock Support**
- Unit tests include mock operators for environments where real operators aren't available
- Graceful fallback ensures tests can run in various environments

### Example Test Usage

```python
def test_streaming_client_op_init_basic(self, fragment):
    """Test basic StreamingClientOp initialization and properties."""
    name = "streaming_client_op"
    op = StreamingClientOp(fragment=fragment, name=name)
    
    assert isinstance(op, BaseOperator)
    assert op.operator_type == Operator.OperatorType.NATIVE
    assert f"name: {name}" in repr(op)
```

### Dependencies

- `pytest`: Testing framework
- `holoscan.core`: Holoscan framework components
- `holohub.streaming_client_operator`: StreamingClient operator module

### Notes

- Tests focus on **unit testing** (operator creation/validation) and **integration testing** (streaming workflows)
- **Network testing** requires server infrastructure and is typically done in integration environments
- Tests are designed to work with standard HoloHub fixtures from `conftest.py`

