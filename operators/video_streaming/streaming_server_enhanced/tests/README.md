# C++ Unit Tests for Streaming Server Operators

This directory contains C++ unit tests for the streaming server operators using the Google Test (GTest) framework.

## Overview

The unit tests verify the correct behavior of three server components:
1. **StreamingServerResource** - Manages server connections and lifecycle
2. **StreamingServerUpstreamOp** - Receives video frames from clients
3. **StreamingServerDownstreamOp** - Sends video frames to clients

Tests focus on:

- ✅ **Resource initialization** with various configurations
- ✅ **Operator creation** with shared resources
- ✅ **Parameter validation** for video resolutions, frame rates, and ports
- ✅ **Streaming direction configuration** (upstream, downstream, bidirectional)
- ✅ **Resource sharing** between multiple operators
- ✅ **Resource management** and cleanup

## Test Coverage

### StreamingServerResource Tests

1. **Basic Resource Tests**
   - `BasicInitialization`: Default resource creation
   - `CustomConfiguration`: Custom port, resolution, FPS
   - `StreamingDirectionConfiguration`: Upstream/downstream combinations
   - `MultiInstanceConfiguration`: Multi-instance mode

2. **Configuration Tests**
   - `VariousResolutions`: SD, HD, Full HD, 4K
   - `VariousFrameRates`: 15, 30, 60, 120 FPS
   - `VariousPortNumbers`: Default, custom, high ports

3. **Resource Management**
   - `ResourceCleanup`: Proper destruction

### StreamingServerUpstreamOp Tests

1. **Initialization Tests**
   - `BasicInitialization`: Operator with resource
   - `CustomVideoParameters`: Override resource settings

2. **Operator Management**
   - `OperatorSetup`: Setup() method
   - `OperatorCleanup`: Proper destruction

### StreamingServerDownstreamOp Tests

1. **Initialization Tests**
   - `BasicInitialization`: Operator with resource
   - `CustomVideoParameters`: Override resource settings

2. **Operator Management**
   - `OperatorSetup`: Setup() method
   - `OperatorCleanup`: Proper destruction

### Integrated Server Tests

1. **Resource Sharing Tests**
   - `SharedResourceConfiguration`: Both operators with one resource
   - `MultipleOperatorsSharedResource`: Multiple operators sharing resource

## Prerequisites

- CMake 3.20 or higher
- Holoscan SDK 3.5.0 or higher
- Google Test (automatically found by CMake)
- C++17 compiler

## Building and Running Tests

### Option 1: Via HoloHub Test 

```bash
# From holohub root directory
# ./holohub test automatically builds with -DBUILD_TESTING=ON
./holohub test video_streaming --ctest-options="-R streaming_server_ops_unit_tests -V"
```

### Option 2: Run Test Executable Directly

```bash
# From holohub root directory
cd build/operators/video_streaming/streaming_server_enhanced/tests

# Run all tests
./test_streaming_server_ops

# Run specific test suite
./test_streaming_server_ops --gtest_filter=StreamingServerResourceTest.*

# Run specific test
./test_streaming_server_ops --gtest_filter=StreamingServerResourceTest.BasicInitialization

# Run with verbose output
./test_streaming_server_ops --gtest_verbose
```

## Test Output Example

```
[==========] Running 18 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 9 tests from StreamingServerResourceTest
[ RUN      ] StreamingServerResourceTest.BasicInitialization
[       OK ] StreamingServerResourceTest.BasicInitialization (10 ms)
[ RUN      ] StreamingServerResourceTest.CustomConfiguration
[       OK ] StreamingServerResourceTest.CustomConfiguration (12 ms)
...
[----------] 9 tests from StreamingServerResourceTest (98 ms total)

[----------] 4 tests from StreamingServerUpstreamOpTest
[ RUN      ] StreamingServerUpstreamOpTest.BasicInitialization
[       OK ] StreamingServerUpstreamOpTest.BasicInitialization (15 ms)
...
[----------] 4 tests from StreamingServerUpstreamOpTest (58 ms total)

[----------] 5 tests from StreamingServerDownstreamOpTest
[ RUN      ] StreamingServerDownstreamOpTest.BasicInitialization
[       OK ] StreamingServerDownstreamOpTest.BasicInitialization (14 ms)
...
[----------] 5 tests from StreamingServerDownstreamOpTest (62 ms total)

[==========] 18 tests from 3 test suites ran. (218 ms total)
[  PASSED  ] 18 tests.
```

## Test Structure

### Resource Test Pattern

```cpp
TEST_F(StreamingServerResourceTest, TestName) {
  // Create resource with specific parameters
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "resource_name",
      holoscan::Arg("port") = uint16_t{48010},
      ...
  );

  // Verify creation
  ASSERT_NE(resource_, nullptr);
  EXPECT_EQ(resource_->name(), "resource_name");
}
```

### Operator Test Pattern

```cpp
TEST_F(StreamingServerUpstreamOpTest, TestName) {
  // 1. Create shared resource
  resource_ = fragment_->make_resource<StreamingServerResource>(...);
  
  // 2. Create operator
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "operator_name",
      holoscan::Arg("streaming_server_resource") = resource_
  );

  // 3. Verify
  ASSERT_NE(upstream_op_, nullptr);
}
```

## Adding New Tests

To add new tests:

1. Add test function in `test_streaming_server_ops.cpp`:
```cpp
TEST_F(StreamingServerResourceTest, MyNewTest) {
  // Test implementation
}
```

2. Rebuild:
```bash
make test_streaming_server_ops
```

3. Run the new test:
```bash
./test_streaming_server_ops --gtest_filter=StreamingServerResourceTest.MyNewTest
```

## Debugging Tests

### Run with GDB
```bash
gdb --args ./test_streaming_server_ops \
  --gtest_filter=StreamingServerResourceTest.BasicInitialization
```

### Enable Verbose Output
```bash
./test_streaming_server_ops --gtest_verbose
```

### List All Tests
```bash
./test_streaming_server_ops --gtest_list_tests
```

### Run Specific Test Suite
```bash
# Run all resource tests
./test_streaming_server_ops --gtest_filter=StreamingServerResourceTest.*

# Run all upstream tests
./test_streaming_server_ops --gtest_filter=StreamingServerUpstreamOpTest.*

# Run all downstream tests
./test_streaming_server_ops --gtest_filter=StreamingServerDownstreamOpTest.*
```

## Important Notes

⚠️ **Network-Free Testing**: These tests do NOT start actual streaming servers or require network connections. Tests focus on operator creation and configuration.

⚠️ **Integration Tests**: For end-to-end testing with actual streaming, see the integration tests in `applications/video_streaming/TESTING.md`.

⚠️ **Shared Resources**: Tests verify that multiple operators can share the same `StreamingServerResource` instance, which is the typical usage pattern in applications.

## Test Organization

Tests are organized by component:

1. **StreamingServerResourceTest**: Resource creation and configuration
2. **StreamingServerUpstreamOpTest**: Upstream operator (receiving frames)
3. **StreamingServerDownstreamOpTest**: Downstream operator (sending frames)
4. **Integrated Tests**: Multiple operators sharing resources

## Architecture Patterns Tested

### Pattern 1: Simple Server (Upstream Only)

```cpp
resource = make_resource<StreamingServerResource>(
    "resource",
    Arg("enable_upstream") = true,
    Arg("enable_downstream") = false
);

upstream_op = make_operator<StreamingServerUpstreamOp>(
    "upstream",
    Arg("streaming_server_resource") = resource
);
```

### Pattern 2: Echo Server (Bidirectional)

```cpp
resource = make_resource<StreamingServerResource>(
    "resource",
    Arg("enable_upstream") = true,
    Arg("enable_downstream") = true
);

upstream_op = make_operator<StreamingServerUpstreamOp>(
    "upstream",
    Arg("streaming_server_resource") = resource
);

downstream_op = make_operator<StreamingServerDownstreamOp>(
    "downstream",
    Arg("streaming_server_resource") = resource
);

// Connect: upstream → downstream
add_flow(upstream_op, downstream_op, {{"output_frames", "input_frames"}});
```

## Differences from PR #1134

**Current Implementation** (This Branch):
- ✅ **C++ Unit Tests**: Full GTest-based tests for all three components
- ✅ **Comprehensive Coverage**: Resource, upstream, downstream, integration tests
- ✅ **Resource Sharing Tests**: Verify shared resource patterns
- ✅ **Well-Documented**: Detailed README with examples
- ✅ **Integrated Build**: Automatic compilation with `-DBUILD_TESTING=ON`

**PR #1134 Implementation**:
- ✅ Had similar C++ unit test structure
- ✅ Had pytest tests for Python bindings
- ❌ Tests were later removed (commit `173b6ee0`)
- ℹ️ Focused on application-level testing

**Key Improvements**:
1. Tests all three server components systematically
2. Tests resource sharing patterns
3. Better organized test suites
4. More comprehensive parameter validation
5. Clearer documentation and examples

## See Also

- **[Streaming Server Operators README](../README.md)** - Operator documentation
- **[Streaming Client Tests](../../streaming_client_enhanced/tests/README.md)** - Client operator tests
- **[Integration Tests](../../../../applications/video_streaming/TESTING.md)** - End-to-end testing
- **[Main README](../../../../applications/video_streaming/README.md)** - Application overview

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```bash
# CI-friendly command
timeout 120 ctest -R streaming_server_ops_unit_tests --output-on-failure
echo "Test exit code: $?"
```

**Exit Codes:**
- `0`: All tests passed
- `1`: One or more tests failed
- `124`: Test timeout (2 minutes)

## Contributing

When adding new tests:
1. Use appropriate test fixture (`StreamingServerResourceTest`, `StreamingServerUpstreamOpTest`, or `StreamingServerDownstreamOpTest`)
2. Follow the existing test naming pattern
3. Use descriptive test names
4. Add documentation comments
5. Update this README if adding new test categories
6. Ensure tests pass before committing

## Running All Video Streaming Tests

To run all video streaming tests (client + server + integration):

```bash
# From holohub root directory
./holohub test video_streaming \
  --cmake-options='-DBUILD_TESTING=ON' \
  --ctest-options="-V"
```

This will run:
- ✅ `streaming_client_op_unit_tests` - Client unit tests
- ✅ `streaming_server_ops_unit_tests` - Server unit tests  
- ✅ `video_streaming_integration_test` - C++ integration test
- ✅ `video_streaming_integration_test_python` - Python integration test (if Python enabled)

