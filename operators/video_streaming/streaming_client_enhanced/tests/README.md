# C++ Unit Tests for StreamingClientOp

This directory contains C++ unit tests for the `StreamingClientOp` operator using the Google Test (GTest) framework.

## Overview

The unit tests verify the correct behavior of the StreamingClientOp operator in isolation, without requiring an actual streaming server connection. Tests focus on:

- ✅ **Operator initialization** with various parameter combinations
- ✅ **Parameter validation** for video resolutions, frame rates, and network settings
- ✅ **Streaming mode configuration** (send-only, receive-only, bidirectional)
- ✅ **Frame validation parameters** (min_non_zero_bytes threshold)
- ✅ **Resource management** and cleanup
- ✅ **Edge cases and boundary conditions**

## Test Coverage

### Test Categories

1. **Basic Initialization Tests**
   - `BasicInitialization`: Standard operator creation
   - `InitializationWithStreamingDisabled`: Configuration-only mode

2. **Parameter Validation Tests**
   - `VideoResolutionParameters`: SD, HD, Full HD, 4K resolutions
   - `FrameRateParameters`: 15, 30, 60, 120 FPS
   - `NetworkParameters`: Various server IPs and ports
   - `StreamingModeParameters`: Send-only, receive-only, bidirectional, disabled
   - `FrameValidationParameter`: Various validation thresholds

3. **Operator Setup Tests**
   - `OperatorSetup`: Verify setup() method

4. **Edge Case and Boundary Tests**
   - `MinimumResolution`: 320x240 resolution
   - `MaximumResolution`: 8K (7680x4320) resolution
   - `PortNumberEdgeCases`: Low and high port numbers

5. **Resource Management Tests**
   - `OperatorCleanup`: Proper destruction
   - `MultipleInstances`: Multiple operators simultaneously

## Prerequisites

- CMake 3.20 or higher
- Holoscan SDK 3.5.0 or higher
- Google Test (automatically found by CMake)
- C++17 compiler

## Building and Running Tests

### Option 1: Via HoloHub Build System

```bash
# From holohub root directory
# ./holohub test automatically builds with -DBUILD_TESTING=ON
./holohub test video_streaming --ctest-options="-R streaming_client_op_unit_tests -V"
```

## Test Output Example

```
[==========] Running 16 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 16 tests from StreamingClientOpTest
[ RUN      ] StreamingClientOpTest.BasicInitialization
[       OK ] StreamingClientOpTest.BasicInitialization (12 ms)
[ RUN      ] StreamingClientOpTest.VideoResolutionParameters
[       OK ] StreamingClientOpTest.VideoResolutionParameters (45 ms)
...
[----------] 16 tests from StreamingClientOpTest (234 ms total)

[==========] 16 tests from 1 test suite ran. (234 ms total)
[  PASSED  ] 16 tests.
```

## Test Structure

Each test follows this pattern:

```cpp
TEST_F(StreamingClientOpTest, TestName) {
  // 1. Create operator with specific parameters
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "operator_name",
      holoscan::Arg("width") = 640u,
      holoscan::Arg("height") = 480u,
      ...
  );

  // 2. Verify creation
  ASSERT_NE(streaming_client_op_, nullptr);
  
  // 3. Verify properties
  EXPECT_EQ(streaming_client_op_->name(), "operator_name");
  ...
}
```

## Adding New Tests

To add new tests:

1. Add test function in `test_streaming_client_op.cpp`:
```cpp
TEST_F(StreamingClientOpTest, MyNewTest) {
  // Test implementation
}
```

2. Rebuild:
```bash
make test_streaming_client_op
```

3. Run the new test:
```bash
./test_streaming_client_op --gtest_filter=StreamingClientOpTest.MyNewTest
```

## Important Notes

⚠️ **Network-Free Testing**: These tests do NOT require an actual streaming server. All tests use `send_frames=false` and `receive_frames=false` to test operator creation and configuration without network operations.

⚠️ **Integration Tests**: For end-to-end testing with actual streaming, see the integration tests in `applications/video_streaming/TESTING.md`.


**Key Improvements**:
1. Tests focus on operator functionality, not just binding validation
2. Better organized test categories
3. More comprehensive parameter validation
4. Clearer documentation
5. Integrated into current build system

## See Also

- **[Streaming Client Operator README](../README.md)** - Operator documentation
- **[Streaming Server Tests](../../streaming_server_enhanced/tests/README.md)** - Server operator tests  
- **[Integration Tests](../../../../applications/video_streaming/TESTING.md)** - End-to-end testing
- **[Main README](../../../../applications/video_streaming/README.md)** - Application overview

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```bash
# CI-friendly command
timeout 120 ctest -R streaming_client_op_unit_tests --output-on-failure
echo "Test exit code: $?"
```

**Exit Codes:**
- `0`: All tests passed
- `1`: One or more tests failed
- `124`: Test timeout (2 minutes)

## Contributing

When adding new tests:
1. Follow the existing test naming pattern
2. Use descriptive test names
3. Add documentation comments
4. Update this README if adding new test categories
5. Ensure tests pass before committing

