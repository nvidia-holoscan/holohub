# StreamingClient Operator Testing

## Unit Tests

This directory contains comprehensive unit tests for the `StreamingClientOp` Python operator.

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

#### Unit Tests:
```bash
# From repository root
pytest operators/streaming_client/test_streaming_client_op.py -v

# From StreamingClient directory
cd operators/streaming_client/
pytest test_streaming_client_op.py -v
```

#### Integration Tests:
```bash
# Python infrastructure test
bash operators/streaming_client/tests/run_test.sh \
  operators/streaming_client/python/streaming_client_demo.py

# Python functional test (requires video data)
bash operators/streaming_client/tests/run_functional_test.sh \
  build/operators/streaming_client/python \
  operators/streaming_client/tests/video_streaming_client_functional.py \
  data/endoscopy

# C++ test (requires built streaming_client_demo)
bash operators/streaming_client/tests/run_cpp_test.sh \
  build/applications/video_streaming/video_streaming_client/cpp/streaming_client_demo \
  operators/streaming_client/testing/video_streaming_client_testing.yaml \
  data/endoscopy
```

### Test Categories

| Category | Test Count | Purpose |
|----------|------------|---------|
| **Unit Tests** | 20+ tests | Operator creation, parameter validation, structure |
| **Infrastructure Tests** | 2 tests | Basic functionality without video data |
| **Functional Tests** | 2 tests | End-to-end streaming with real video data |
| **Golden Frame Tests** | 1 test | Visual regression testing |

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
- All tests follow HoloHub testing best practices and conventions
