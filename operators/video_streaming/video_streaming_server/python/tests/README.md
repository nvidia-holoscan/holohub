# Python Unit Tests for StreamingServer Operators Bindings

This directory contains Python unit tests for the StreamingServer operators Python bindings (pybind11).

> **📖 See Also:** For general information about pytest setup, prerequisites, CUDA compatibility, and troubleshooting, refer to **[PYTEST_SUMMARY.md](../../../PYTEST_SUMMARY.md)**.

## Overview

The pytest tests verify the Python bindings of the StreamingServer operators, focusing on:

- ✅ **StreamingServerResource** - Server resource management
- ✅ **StreamingServerUpstreamOp** - Frame reception operator
- ✅ **StreamingServerDownstreamOp** - Frame transmission operator
- ✅ **Resource sharing** - Multiple operators with shared resource
- ✅ **Python/C++ binding correctness** - Proper exposure to Python
- ✅ **Parameter handling** - Cross-language parameter passing
- ✅ **Memory management** - No leaks across boundaries

**Test Count:** 51 test executions (42 test functions + 9 parametrized variants)

## Test Coverage

### TestStreamingServerResourceBinding

**Basic Tests:**
- `test_resource_creation_basic` - Basic resource instantiation
- `test_resource_name` - Name property validation
- `test_resource_inheritance` - Inheritance from holoscan.Resource

**Parametrized Tests:**
- `test_video_parameters` - Various resolutions and frame rates
- `test_port_parameters` - Different port numbers
- `test_streaming_direction` - Upstream/downstream combinations
- `test_server_name_parameter` - Server naming

**Advanced Tests:**
- `test_memory_management` - Multiple resources, garbage collection
- `test_multiple_resources_different_ports` - Port isolation

### TestStreamingServerUpstreamOpBinding

**Basic Tests:**
- `test_operator_creation_basic` - Basic operator instantiation
- `test_operator_name` - Name property validation
- `test_operator_inheritance` - Inheritance from holoscan.Operator
- `test_method_availability` - Required methods are exposed

**Resource Tests:**
- `test_operator_with_custom_resource` - Custom resource configuration
- `test_multiple_operators_shared_resource` - Resource sharing

### TestStreamingServerDownstreamOpBinding

**Basic Tests:**
- `test_operator_creation_basic` - Basic operator instantiation
- `test_operator_name` - Name property validation
- `test_operator_inheritance` - Inheritance from holoscan.Operator
- `test_method_availability` - Required methods are exposed

**Resource Tests:**
- `test_operator_with_custom_resource` - Custom resource configuration
- `test_multiple_operators_shared_resource` - Resource sharing

### TestStreamingServerIntegration

**Integration Tests:**
- `test_bidirectional_server_setup` - Upstream + downstream operators
- `test_multiple_servers_different_ports` - Multiple server instances
- `test_operators_in_application_context` - Works within Application
- `test_resource_isolation_between_fragments` - Fragment isolation

### TestStreamingServerWithMockData

**Mock Data Tests:**
- `test_resource_with_mock_frame_dimensions` - Resource with various dimensions
- `test_upstream_operator_with_mock_frames` - Upstream with mock data
- `test_downstream_operator_with_mock_frames` - Downstream with mock data
- `test_bidirectional_server_with_mock_frames` - Bidirectional with mock data
- `test_multiple_resolutions_with_mock_frames` - Multiple resolutions
- `test_server_with_numpy_and_cupy_frames` - NumPy/CuPy compatibility
- `test_server_with_float_frames` - Float32 data type

### TestStreamingServerUpstreamOpCompute

**Upstream Compute Tests:**
- `test_compute_method_exists` - Method accessibility
- `test_upstream_compute_with_mock_frame` - Basic compute functionality
- `test_upstream_compute_with_various_resolutions` - Multiple resolutions
- `test_upstream_compute_method_signature` - Method signature validation

### TestStreamingServerDownstreamOpCompute

**Downstream Compute Tests:**
- `test_compute_method_exists` - Method accessibility
- `test_downstream_compute_with_mock_frame` - Basic compute functionality
- `test_downstream_compute_with_various_resolutions` - Multiple resolutions
- `test_downstream_compute_with_float_frames` - Float32 frames
- `test_downstream_compute_method_signature` - Method signature validation

### TestBidirectionalServerCompute

**Bidirectional Compute Tests:**
- `test_bidirectional_compute_flow` - Full bidirectional compute flow

## Running Tests

### Option 1: Via CMake/CTest (Recommended)

```bash
# From holohub root
# ./holohub test automatically builds with -DBUILD_TESTING=ON and compiles Python bindings
./holohub test video_streaming --cuda 12 --ctest-options="-R video_streaming_server_pytest -VV"
```

### Option 2: Direct pytest Inside Container

For interactive testing and debugging, you can run pytest directly inside the holohub container.

> **⚠️ Important:** You must run `./holohub test` first (Option 1) to build the Python bindings.

**Step 1: Build Python bindings by running tests once**

```bash
# From holohub root - this builds Python bindings and runs tests
./holohub test video_streaming --cuda 12 --ctest-options="-R video_streaming_server_pytest -VV"
```

**Step 2: Launch container and run pytest**

```bash
# Launch interactive container with bash
./holohub run-container video_streaming --cuda 12 --docker-opts='--entrypoint=bash'

# Inside container, run all server tests
pytest operators/video_streaming/video_streaming_server/python/tests/ -v

# Inside container, run specific test class
pytest operators/video_streaming/video_streaming_server/python/tests/test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding -v

# Inside container, run specific test method
pytest operators/video_streaming/video_streaming_server/python/tests/test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_resource_creation_basic -v
```

> **Note:** Direct pytest may encounter segfaults in compute tests. Option 1 (CTest) is more reliable as it runs tests in isolation.

## Test Output Example

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 51 items

test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_resource_creation_basic PASSED [  2%]
test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_resource_name PASSED [  5%]
test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_video_parameters[640-480-30] PASSED [  8%]
...
test_streaming_server_ops_bindings.py::TestStreamingServerWithMockData::test_server_with_float_frames PASSED [ 90%]
test_streaming_server_ops_bindings.py::TestStreamingServerUpstreamOpCompute::test_compute_method_exists PASSED [ 95%]
test_streaming_server_ops_bindings.py::TestBidirectionalServerCompute::test_bidirectional_compute_flow PASSED [100%]

========================== 51 passed in 3.2s ==================================
```

## Fixtures

Defined in `conftest.py`:

**Module Fixtures:**
- `streaming_server_module` - VideoStreamingServer Python module
- `streaming_server_classes` - All server operator classes (Resource, Upstream, Downstream)

**Factory Fixtures:**
- `resource_factory` - Factory for creating StreamingServerResource
- `upstream_operator_factory` - Factory for creating upstream operators
- `downstream_operator_factory` - Factory for creating downstream operators

**Default Fixtures:**
- `default_resource` - Pre-configured resource with defaults

**Common Fixtures** (copied from root conftest.py):
- `app` - Holoscan Application instance
- `fragment` - Holoscan Fragment for testing
- `mock_image` - Factory for creating mock image tensors
- `op_input_factory` - Factory for mock operator inputs
- `op_output` - Mock operator output
- `execution_context` - Mock execution context

> **Note:** Common fixtures are duplicated locally because `add_python_tests()` uses `--confcutdir` for test isolation, preventing access to the root conftest.py.

## Writing New Tests

### Testing Resources

```python
def test_custom_resource(self, resource_factory):
    """Test custom resource configuration."""
    resource = resource_factory(
        name="custom",
        port=9000,
        width=1920,
        height=1080
    )
    assert resource is not None
```

### Testing Operators

```python
def test_custom_operator(self, upstream_operator_factory, default_resource):
    """Test operator with custom configuration."""
    op = upstream_operator_factory(
        name="custom_op",
        resource=default_resource
    )
    assert op is not None
```

### Testing Resource Sharing

```python
def test_shared_resource(
    self,
    resource_factory,
    upstream_operator_factory,
    downstream_operator_factory
):
    """Test operators sharing a resource."""
    resource = resource_factory()
    upstream = upstream_operator_factory(resource=resource)
    downstream = downstream_operator_factory(resource=resource)
    
    assert upstream is not None
    assert downstream is not None
```

## Test Organization

```
python/tests/
├── conftest.py                                 # Pytest fixtures
├── test_streaming_server_ops_bindings.py       # Main test file
└── README.md                                   # This file
```

## Related Documentation

- **[Python Tests Summary (PYTEST_SUMMARY.md)](../../../PYTEST_SUMMARY.md)** - Overview of all Python binding tests
- [C++ Unit Tests](../../tests/README.md) - C++ operator unit tests
- [StreamingServerOps Documentation](../../README.md) - Server operator documentation
- [Integration Tests](../../../../../applications/video_streaming/TESTING.md) - End-to-end tests
