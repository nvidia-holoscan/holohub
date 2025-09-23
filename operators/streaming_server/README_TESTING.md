# StreamingServer Operator Testing

## Unit Tests

This directory contains comprehensive unit tests for the `StreamingServerOp` Python operator.

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

#### From Repository Root:
```bash
# Run all StreamingServer tests
pytest operators/streaming_server/test_streaming_server_op.py -v

# Run with coverage
pytest --cov=operators/streaming_server/ operators/streaming_server/test_streaming_server_op.py

# Run specific test category
pytest operators/streaming_server/test_streaming_server_op.py::TestStreamingServerOp::test_streaming_server_op_init_basic -v
```

#### From StreamingServer Directory:
```bash
cd operators/streaming_server/
pytest test_streaming_server_op.py -v
```

### Test Categories

| Category | Test Count | Purpose |
|----------|------------|---------|
| **Initialization** | 8 tests | Operator creation with various parameters |
| **Port Setup** | 1 test | Validates standalone architecture (no ports) |
| **Error Handling** | 6 tests | Invalid parameter handling |
| **Edge Cases** | 6 tests | Boundary conditions and special cases |
| **Architecture** | 4 tests | Operator structure and lifecycle |

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
- All tests follow HoloHub testing best practices and conventions
