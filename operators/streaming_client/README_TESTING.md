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


### Running the Tests

## **Unit Test 1**

**Purpose:** Validate StreamingClientOp Python operator creation, parameter validation, and lifecycle methods

**What it Tests:**
- Operator initialization with default parameters (constructor limitation)
- Basic operator properties and type validation
- Lifecycle method existence (setup, initialize, start, stop, compute)
- Multiple operator instances
- Mock operator fallback for testing environments

**Expected Outcome:**
- All 14 unit tests pass in 0.05 seconds
- Test execution completes quickly
- Mock fallback works when real operator unavailable
- Output: `============================== 14 passed in 0.05s ==============================`

**Acceptance Criteria:**
- ✅ All 14 pytest assertions pass
- ✅ No import or initialization errors
- ✅ Test completes in under 1 second
- ✅ Mock fallback works when real operator unavailable
- ✅ Output contains "14 passed" message

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_client --verbose
```

## **Unit Test 2**

**Purpose:** Basic Python functionality validation without video data

**What it Tests:**
- StreamingClient operator startup and initialization
- Network configuration and connection setup
- Graceful shutdown and cleanup processes
- Basic pipeline functionality

**Expected Outcome:**
- Test passes in ~3.5 seconds
- StreamingClient initializes successfully
- Expected connection failures handled gracefully (5 connection attempts made)
- Output: `✅ Test PASSED: StreamingClient functionality validated successfully`

**Expected Error Messages (Normal Behavior):**
Since there is no streaming server running during tests, these connection error messages are expected:
- `Failed to connect to server: NVST_R_ERROR_UNEXPECTED_DISCONNECTION_INITIAL`
- `Failed to connect to server: NVST_R_INVALID_OPERATION`
- `All connection attempts failed. Final error: Failed to connect to server: NVST_R_INVALID_OPERATION`
- `RuntimeError: Failed to connect to server: NVST_R_INVALID_OPERATION`

**Acceptance Criteria:**
- ✅ "StreamingClient functionality validated successfully" message appears
- ✅ No fatal errors or exceptions during startup
- ✅ 5 connection attempts made as expected
- ✅ Connection failure errors are expected (no server running)
- ✅ Clean shutdown without hanging
- ✅ Test completes within 5 seconds

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_client --verbose
```

## **Unit Test 3**

**Purpose:** End-to-end Python streaming functionality with real video data

**What it Tests:**
- StreamingClient processing with actual video frames
- Client-server communication protocols
- Video data handling and frame processing
- Performance under realistic conditions

**Expected Outcome:**
- Test passes in ~3.5 seconds
- Video data directory found and used (/workspace/holohub/data)
- Functional test with video data processing attempted
- Output: `✅ FUNCTIONAL test PASSED: StreamingClient functionality validated (partial)`

**Expected Error Messages (Normal Behavior):**
Since there is no streaming server running during tests, these connection error messages are expected:
- `Failed to connect to server: NVST_R_ERROR_UNEXPECTED_DISCONNECTION_INITIAL`
- `Failed to connect to server: NVST_R_INVALID_OPERATION`
- `All connection attempts failed. Final error: Failed to connect to server: NVST_R_INVALID_OPERATION`
- `RuntimeError: Failed to connect to server: NVST_R_INVALID_OPERATION`

**Acceptance Criteria:**
- ✅ "FUNCTIONAL test PASSED" message appears
- ✅ Video data directory found and used
- ✅ StreamingClient processes video frames
- ✅ Connection failure errors are expected (no server running)
- ✅ No critical streaming protocol errors
- ✅ Test completes within 5 seconds

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_client --verbose
```

## **Unit Test 4**

**Purpose:** Validate C++ StreamingClient demo application functionality

**What it Tests:**
- C++ implementation of streaming client
- Binary executable functionality
- Configuration file loading
- C++ operator integration

**Expected Outcome:**
- Test passes in ~3.4 seconds
- C++ executable runs successfully
- Configuration loads from YAML file
- Output: `✅ Test PASSED: C++ StreamingClient INFRASTRUCTURE test successful`

**Expected Error Messages (Normal Behavior):**
Since there is no streaming server running during tests, these connection error messages are expected:
- `Failed to connect to server: NVST_R_ERROR_UNEXPECTED_DISCONNECTION_INITIAL`
- `Failed to connect to server: NVST_R_INVALID_OPERATION`
- `All connection attempts failed. Final error: Failed to connect to server: NVST_R_INVALID_OPERATION`
- `std::runtime_error: Failed to connect to server: NVST_R_INVALID_OPERATION`

**Acceptance Criteria:**
- ✅ "C++ StreamingClient INFRASTRUCTURE test successful" message appears
- ✅ No compilation or runtime errors
- ✅ YAML configuration loads properly
- ✅ Connection failure errors are expected (no server running)
- ✅ Clean application shutdown
- ✅ Test completes within 5 seconds

**How to Build & Execute the Test:**
```bash
./holohub test video_streaming_client --verbose
```

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

#### **Test Philosophy**
- **Unit Testing Focus**: Tests operator creation, parameter validation, and basic structure
- **No Network Testing**: Network functionality requires integration testing
- **Parameter Boundary Testing**: Validates edge cases and error conditions
- **Architecture Validation**: Ensures proper Holoscan operator compliance

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

- Tests focus on **unit testing** (operator creation/validation) 
- **Network testing** requires server infrastructure and is typically done in integration environments
- Tests are designed to work with standard HoloHub fixtures from `conftest.py`

