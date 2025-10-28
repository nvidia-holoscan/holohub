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
./holohub test video_streaming --ctest-options="-R streaming_server_ops_unit_tests -VV"
```

## Test Output Example

```text
[==========] Running 18 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 8 tests from StreamingServerResourceTest
[ RUN      ] StreamingServerResourceTest.BasicInitialization
[       OK ] StreamingServerResourceTest.BasicInitialization (0 ms)
[ RUN      ] StreamingServerResourceTest.CustomConfiguration
[       OK ] StreamingServerResourceTest.CustomConfiguration (0 ms)
[ RUN      ] StreamingServerResourceTest.StreamingDirectionConfiguration
[       OK ] StreamingServerResourceTest.StreamingDirectionConfiguration (0 ms)
[ RUN      ] StreamingServerResourceTest.MultiInstanceConfiguration
[       OK ] StreamingServerResourceTest.MultiInstanceConfiguration (0 ms)
[ RUN      ] StreamingServerResourceTest.VariousResolutions
[       OK ] StreamingServerResourceTest.VariousResolutions (0 ms)
[ RUN      ] StreamingServerResourceTest.VariousFrameRates
[       OK ] StreamingServerResourceTest.VariousFrameRates (0 ms)
[ RUN      ] StreamingServerResourceTest.VariousPortNumbers
[       OK ] StreamingServerResourceTest.VariousPortNumbers (0 ms)
[ RUN      ] StreamingServerResourceTest.ResourceCleanup
[       OK ] StreamingServerResourceTest.ResourceCleanup (0 ms)
[----------] 8 tests from StreamingServerResourceTest (0 ms total)

[----------] 6 tests from StreamingServerUpstreamOpTest
[ RUN      ] StreamingServerUpstreamOpTest.BasicInitialization
[info] [streaming_server_upstream_op.cpp:72] StreamingServerUpstreamOp setup completed - receives frames from clients
[info] [streaming_server_upstream_op.cpp:46] StreamingServerUpstreamOp destructor: beginning cleanup...
[info] [streaming_server_upstream_op.cpp:51] StreamingServerUpstreamOp destructor: cleanup completed
[       OK ] StreamingServerUpstreamOpTest.BasicInitialization (0 ms)
[ RUN      ] StreamingServerUpstreamOpTest.CustomVideoParameters
[info] [streaming_server_upstream_op.cpp:72] StreamingServerUpstreamOp setup completed - receives frames from clients
[info] [streaming_server_upstream_op.cpp:46] StreamingServerUpstreamOp destructor: beginning cleanup...
[info] [streaming_server_upstream_op.cpp:51] StreamingServerUpstreamOp destructor: cleanup completed
[       OK ] StreamingServerUpstreamOpTest.CustomVideoParameters (0 ms)
[ RUN      ] StreamingServerUpstreamOpTest.OperatorSetup
[ RUN      ] StreamingServerUpstreamOpTest.OperatorCleanup
[ RUN      ] StreamingServerUpstreamOpTest.SharedResourceConfiguration
[info] [streaming_server_downstream_op.cpp:69] StreamingServerDownstreamOp setup completed - sends frames to clients
[info] [streaming_server_downstream_op.cpp:43] StreamingServerDownstreamOp destructor: beginning cleanup...
[info] [streaming_server_downstream_op.cpp:48] StreamingServerDownstreamOp destructor: cleanup completed
[       OK ] StreamingServerUpstreamOpTest.SharedResourceConfiguration (0 ms)
[ RUN      ] StreamingServerUpstreamOpTest.MultipleOperatorsSharedResource
[info] [streaming_server_upstream_op.cpp:72] StreamingServerUpstreamOp setup completed - receives frames from clients
[info] [streaming_server_downstream_op.cpp:69] StreamingServerDownstreamOp setup completed - sends frames to clients
[info] [streaming_server_downstream_op.cpp:43] StreamingServerDownstreamOp destructor: beginning cleanup...
[info] [streaming_server_downstream_op.cpp:48] StreamingServerDownstreamOp destructor: cleanup completed
[       OK ] StreamingServerUpstreamOpTest.MultipleOperatorsSharedResource (0 ms)
[----------] 6 tests from StreamingServerUpstreamOpTest (0 ms total)

[----------] 4 tests from StreamingServerDownstreamOpTest
[ RUN      ] StreamingServerDownstreamOpTest.BasicInitialization
[info] [streaming_server_downstream_op.cpp:69] StreamingServerDownstreamOp setup completed - sends frames to clients
[info] [streaming_server_downstream_op.cpp:43] StreamingServerDownstreamOp destructor: beginning cleanup...
[info] [streaming_server_downstream_op.cpp:48] StreamingServerDownstreamOp destructor: cleanup completed
[       OK ] StreamingServerDownstreamOpTest.BasicInitialization (0 ms)
[ RUN      ] StreamingServerDownstreamOpTest.CustomVideoParameters
[info] [streaming_server_downstream_op.cpp:69] StreamingServerDownstreamOp setup completed - sends frames to clients
[info] [streaming_server_downstream_op.cpp:43] StreamingServerDownstreamOp destructor: beginning cleanup...
[info] [streaming_server_downstream_op.cpp:48] StreamingServerDownstreamOp destructor: cleanup completed
[       OK ] StreamingServerDownstreamOpTest.CustomVideoParameters (0 ms)
[ RUN      ] StreamingServerDownstreamOpTest.OperatorSetup
[ RUN      ] StreamingServerDownstreamOpTest.OperatorCleanup
[----------] 4 tests from StreamingServerDownstreamOpTest (0 ms total)

[----------] Global test environment tear-down
[==========] 18 tests from 3 test suites ran. (0 ms total)
[  PASSED  ] 18 tests.

100% tests passed, 0 tests failed out of 1

Label Time Summary:
streaming_server    =   0.06 sec*proc (1 test)
unit                =   0.06 sec*proc (1 test)

Total Test time (real) =   0.06 sec
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

## See Also

- **[Streaming Server Operators README](../README.md)** - Operator documentation
- **[Streaming Client Tests](../../streaming_client_enhanced/tests/README.md)** - Client operator tests
- **[Integration Tests](../../../../applications/video_streaming/TESTING.md)** - End-to-end testing
- **[Main README](../../../../applications/video_streaming/README.md)** - Application overview
