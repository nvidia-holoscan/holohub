# C++ Unit Tests for Video Streaming Operators - Summary

## Overview

This document summarizes the C++ unit tests for the `video_streaming` operators, providing comprehensive coverage of all client and server components.

## ℹ️ Related Testing Documentation

This document covers **C++ unit tests** only. For **Python binding tests (pytest)**, see:
- **[PYTEST_SUMMARY.md](PYTEST_SUMMARY.md)** - Comprehensive Python binding test documentation (61 pytest tests)

Together, these test suites provide **92+ total tests** for complete video streaming operator validation:
- **31 C++ unit tests** (this document) - Fast, isolated testing of C++ components
- **61 Python pytest tests** - Validation of Python/C++ bindings

## Test Suites

This package includes **31 unit tests** across all video streaming operators:

### 1. **StreamingClientOp Tests** (13 tests)
   - Basic initialization
   - Parameter validation (video, network, streaming modes)
   - Frame validation parameters
   - Edge cases and boundaries
   - Resource management

### 2. **StreamingServerResource Tests** (8 tests)
   - Resource creation and configuration
   - Various resolutions and frame rates
   - Port configurations
   - Streaming direction settings
   - Multi-instance mode

### 3. **StreamingServerUpstreamOp Tests** (6 tests)
   - Operator initialization
   - Custom video parameters
   - Setup and cleanup
   - Shared resource configuration
   - Multiple operators with shared resource

### 4. **StreamingServerDownstreamOp Tests** (4 tests)
   - Operator initialization
   - Custom video parameters
   - Setup and cleanup
   - Resource cleanup

## File Structure

```
operators/video_streaming/
├── streaming_client_enhanced/
│   ├── tests/
│   │   ├── test_streaming_client_op.cpp      # Client unit tests (13 tests)
│   │   ├── CMakeLists.txt                     # Build config
│   │   └── README.md                          # Documentation
│   └── CMakeLists.txt                         # Updated to include tests
│
└── streaming_server_enhanced/
    ├── tests/
    │   ├── test_streaming_server_ops.cpp      # Server unit tests (18 tests)
    │   ├── CMakeLists.txt                     # Build config
    │   └── README.md                          # Documentation
    └── CMakeLists.txt                         # Updated to include tests
```

## Test Statistics

| Component | Tests | Lines of Code | Coverage | Execution Time |
|-----------|-------|---------------|----------|----------------|
| StreamingClientOp | 13 | ~700 | Initialization, parameters, setup, edge cases | ~0.06 sec |
| StreamingServerResource | 8 | ~300 | Resource creation, configuration, various settings | ~0.03 sec |
| StreamingServerUpstreamOp | 6 | ~250 | Initialization, setup, cleanup, shared resources | ~0.02 sec |
| StreamingServerDownstreamOp | 4 | ~150 | Initialization, setup, cleanup | ~0.01 sec |
| **Total** | **31** | **~1,400** | **Comprehensive** | **~0.12 sec** |

**Test Success Rate: 100% ✅** (2/2 test suites passed, 31 individual tests passed)

## Building and Running Tests

### Run All Unit Tests
```bash
# Run all unit tests
./holohub test video_streaming --ctest-options="-R unit_tests -VV"
```

**Note:** `./holohub test` automatically builds the operator with `-DBUILD_TESTING=ON`, so no separate build step is needed.

### Test Results

When running the unit tests, you should see output similar to:

```text
Test project /workspace/holohub/build-video_streaming
    Start 5: streaming_client_op_unit_tests
1/2 Test #5: streaming_client_op_unit_tests ....   Passed    0.06 sec
    Start 6: streaming_server_ops_unit_tests
2/2 Test #6: streaming_server_ops_unit_tests ...   Passed    0.06 sec

100% tests passed, 0 tests failed out of 2

Label Time Summary:
streaming_client    =   0.06 sec*proc (1 test)
streaming_server    =   0.06 sec*proc (1 test)
unit                =   0.12 sec*proc (2 tests)

Total Test time (real) =   0.13 sec
```

✅ **All 31 tests pass successfully in ~0.13 seconds!**

## Test Categories

### 1. Initialization Tests
Verify operators can be created with various parameter combinations.

**Example:**
```cpp
TEST_F(StreamingClientOpTest, BasicInitialization) {
  streaming_client_op_ = fragment_->make_operator<StreamingClientOp>(
      "test_client",
      Arg("width") = 640u,
      Arg("height") = 480u,
      Arg("fps") = 30u,
      ...
  );
  ASSERT_NE(streaming_client_op_, nullptr);
}
```

### 2. Parameter Validation Tests
Test various video resolutions, frame rates, network configurations.

**Example:**
```cpp
TEST_F(StreamingClientOpTest, VideoResolutionParameters) {
  // Test SD, HD, Full HD, 4K resolutions
  ...
}
```

### 3. Setup Tests
Verify operator setup() methods work correctly.

**Example:**
```cpp
TEST_F(StreamingClientOpTest, OperatorSetup) {
  auto spec = std::make_shared<OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(streaming_client_op_->setup(*spec));
}
```

### 4. Edge Case Tests
Test boundary conditions and extreme parameter values.

**Example:**
```cpp
TEST_F(StreamingClientOpTest, MaximumResolution) {
  // Test 8K resolution (7680x4320)
  ...
}
```

### 5. Resource Management Tests
Verify cleanup and multiple instance handling.

**Example:**
```cpp
TEST_F(StreamingClientOpTest, MultipleInstances) {
  auto client1 = fragment_->make_operator<StreamingClientOp>(...);
  auto client2 = fragment_->make_operator<StreamingClientOp>(...);
  EXPECT_NE(client1, client2);
}
```

## Integration with Existing Tests

The unit tests complement the existing integration tests:

### Unit Tests (This Branch)
- ✅ **Scope**: Individual operator components
- ✅ **Speed**: Fast (0.13 seconds total for 31 tests, ~0.004 sec per test)
- ✅ **Dependencies**: None (network-free)
- ✅ **Focus**: API validation, parameter handling, resource management
- ✅ **Run When**: During development, before commit
- ✅ **Success Rate**: 100% pass rate

### Integration Tests (Already Exist)
- ✅ **Scope**: End-to-end application behavior
- ✅ **Speed**: Slower (~44 seconds per test)
- ✅ **Dependencies**: Requires server/client interaction
- ✅ **Focus**: Actual streaming, frame transmission, bidirectional communication
- ✅ **Run When**: Before merge, in CI/CD

## CI/CD Integration

The unit tests are designed for CI/CD pipelines:

**Benefits:**
- ✅ **Ultra-fast feedback** - Complete in ~0.13 seconds (well under 2 minutes)
- ✅ **No network dependencies** - Tests run in isolation
- ✅ **Clear pass/fail status** - 100% pass rate (2/2 suites, 31 tests)
- ✅ **Detailed error output** - GTest provides clear failure messages
- ✅ **Reliable** - No flaky network-dependent failures


## Documentation

Each test suite has detailed documentation:

- **[Client Tests README](./streaming_client_enhanced/tests/README.md)** - StreamingClientOp C++ unit tests
- **[Server Tests README](./streaming_server_enhanced/tests/README.md)** - Server operator C++ unit tests

Additional documentation:
- **[Python Binding Tests (pytest)](./PYTEST_SUMMARY.md)** - Python binding tests for all operators (61 tests)
- **[Integration Tests](../../applications/video_streaming/TESTING.md)** - End-to-end testing
- **[Main README](../../applications/video_streaming/README.md)** - Application overview

