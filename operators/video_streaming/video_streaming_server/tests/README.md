# C++ Unit Tests for Video Streaming Server Operators

This directory contains C++ unit tests for the video streaming server components using the Google Test (GTest) framework.

## Overview

The unit tests verify the correct behavior of the video streaming server operators and resource in isolation, without requiring actual client connections. Tests focus on:

- ✅ **StreamingServerResource initialization** with various configurations
- ✅ **StreamingServerUpstreamOp initialization** and parameter validation
- ✅ **StreamingServerDownstreamOp initialization** and parameter validation
- ✅ **Parameter validation** for video resolutions, frame rates, and network settings
- ✅ **Server configuration** (port, server name, enable/disable upstream/downstream)
- ✅ **Multi-instance configuration** support
- ✅ **Resource management** and cleanup
- ✅ **Edge cases and boundary conditions**

## Test Coverage

### Test Categories

#### StreamingServerResource Tests

1. **Basic Initialization Tests**
   - `BasicResourceInitialization`: Standard resource creation with defaults
   - `ResourceInitializationWithCustomParameters`: Custom port, name, and dimensions

2. **Configuration Tests**
   - `ResourceConfigurationParameters`: Various server configurations
   - `ResourcePortConfiguration`: Different port numbers
   - `ResourceMultiInstanceMode`: Multi-instance server mode
   - `ResourceUpstreamDownstreamConfiguration`: Enable/disable upstream and downstream

3. **Resource Cleanup Tests**
   - `ResourceCleanup`: Proper destruction
   - `MultipleResourceInstances`: Multiple resource instances

#### StreamingServerUpstreamOp Tests

1. **Basic Initialization Tests**
   - `UpstreamOpBasicInitialization`: Standard operator creation
   - `UpstreamOpWithCustomParameters`: Custom video parameters

2. **Parameter Validation Tests**
   - `UpstreamOpVideoResolutionParameters`: Various resolutions
   - `UpstreamOpFrameRateParameters`: Different frame rates

3. **Operator Setup Tests**
   - `UpstreamOpSetup`: Verify setup() method

4. **Resource Cleanup Tests**
   - `UpstreamOpCleanup`: Proper destruction
   - `MultipleUpstreamOpInstances`: Multiple operator instances

#### StreamingServerDownstreamOp Tests

1. **Basic Initialization Tests**
   - `DownstreamOpBasicInitialization`: Standard operator creation
   - `DownstreamOpWithCustomParameters`: Custom video parameters

2. **Parameter Validation Tests**
   - `DownstreamOpVideoResolutionParameters`: Various resolutions
   - `DownstreamOpFrameRateParameters`: Different frame rates

3. **Operator Setup Tests**
   - `DownstreamOpSetup`: Verify setup() method

4. **Resource Cleanup Tests**
   - `DownstreamOpCleanup`: Proper destruction
   - `MultipleDownstreamOpInstances`: Multiple operator instances

#### Integration Tests

1. **Resource and Operator Integration**
   - `ResourceWithUpstreamOp`: Resource shared with upstream operator
   - `ResourceWithDownstreamOp`: Resource shared with downstream operator
   - `ResourceWithBothOps`: Resource shared with both operators

## Prerequisites

- CMake 3.20 or higher
- Holoscan SDK 3.5.0 or higher
- Google Test (automatically fetched by CMake)
- C++17 compiler

## Building and Running Tests

### Via HoloHub Build System

```bash
# From holohub root directory
# ./holohub test automatically builds with -DBUILD_TESTING=ON
./holohub test video_streaming --ctest-options="-R video_streaming_server_ops_unit_tests -VV"
```

## Test Output Example

```text
[==========] Running 25 tests from 4 test suites.
[----------] Global test environment set-up.
[----------] 8 tests from StreamingServerResourceTest
[ RUN      ] StreamingServerResourceTest.BasicResourceInitialization
[       OK ] StreamingServerResourceTest.BasicResourceInitialization (0 ms)
[ RUN      ] StreamingServerResourceTest.ResourceInitializationWithCustomParameters
[       OK ] StreamingServerResourceTest.ResourceInitializationWithCustomParameters (0 ms)
...
[----------] 8 tests from StreamingServerResourceTest (0 ms total)

[----------] 7 tests from StreamingServerUpstreamOpTest
[ RUN      ] StreamingServerUpstreamOpTest.UpstreamOpBasicInitialization
[       OK ] StreamingServerUpstreamOpTest.UpstreamOpBasicInitialization (0 ms)
...
[----------] 7 tests from StreamingServerUpstreamOpTest (0 ms total)

[----------] 7 tests from StreamingServerDownstreamOpTest
[ RUN      ] StreamingServerDownstreamOpTest.DownstreamOpBasicInitialization
[       OK ] StreamingServerDownstreamOpTest.DownstreamOpBasicInitialization (0 ms)
...
[----------] 7 tests from StreamingServerDownstreamOpTest (0 ms total)

[----------] 3 tests from StreamingServerIntegrationTest
[ RUN      ] StreamingServerIntegrationTest.ResourceWithUpstreamOp
[       OK ] StreamingServerIntegrationTest.ResourceWithUpstreamOp (0 ms)
...
[----------] 3 tests from StreamingServerIntegrationTest (0 ms total)

[----------] Global test environment tear-down
[==========] 25 tests from 4 test suites ran. (0 ms total)
[  PASSED  ] 25 tests.

100% tests passed, 0 tests failed out of 1

Label Time Summary:
video_streaming_server    =   0.08 sec*proc (1 test)
unit                      =   0.08 sec*proc (1 test)

Total Test time (real) =   0.09 sec
```

## Test Structure

### Resource Test Pattern

```cpp
TEST_F(StreamingServerResourceTest, TestName) {
  // 1. Create resource with specific parameters
  resource_ = fragment_->make_resource<StreamingServerResource>(
      "test_resource",
      holoscan::Arg("port") = uint16_t{48010},
      holoscan::Arg("width") = 640u,
      ...
  );

  // 2. Verify creation
  ASSERT_NE(resource_, nullptr);
  
  // 3. Verify properties
  EXPECT_EQ(resource_->name(), "test_resource");
}
```

### Operator Test Pattern

```cpp
TEST_F(StreamingServerUpstreamOpTest, TestName) {
  // 1. Create shared resource
  auto resource = fragment_->make_resource<StreamingServerResource>(...);
  
  // 2. Create operator with resource
  upstream_op_ = fragment_->make_operator<StreamingServerUpstreamOp>(
      "test_op",
      holoscan::Arg("video_streaming_server_resource") = resource,
      ...
  );

  // 3. Verify creation
  ASSERT_NE(upstream_op_, nullptr);
}
```

## Important Notes

⚠️ **Network-Free Testing**: These tests do NOT require actual client connections. All tests verify operator and resource creation and configuration without starting the actual streaming server.

⚠️ **Resource Sharing**: The tests verify that StreamingServerResource can be properly shared between multiple operators, which is the intended usage pattern in production.

⚠️ **Integration Tests**: For end-to-end testing with actual streaming connections, see the integration tests in `applications/video_streaming/TESTING.md`.

## Key Features Tested

1. **Operator and Resource Creation**: Verify proper initialization with various parameter combinations
2. **Parameter Validation**: Test different resolutions, frame rates, ports, and configurations
3. **Resource Sharing**: Ensure resource can be shared between upstream and downstream operators
4. **Configuration Flexibility**: Test both default and custom configurations
5. **Cleanup and Destruction**: Verify proper resource cleanup

## See Also

- **[Video Streaming Server README](../README.md)** - Operator documentation
- **[Video Streaming Client Tests](../../video_streaming_client/tests/README.md)** - Client operator tests
- **[Integration Tests](../../../../applications/video_streaming/TESTING.md)** - End-to-end testing
- **[Main README](../../../../applications/video_streaming/README.md)** - Application overview

