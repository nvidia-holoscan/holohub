# Python Unit Tests for StreamingServer Operators Bindings

This directory contains Python unit tests for the StreamingServer operators Python bindings (pybind11).

## Overview

The pytest tests verify the Python bindings of the StreamingServer operators, focusing on:

- ✅ **StreamingServerResource** - Server resource management
- ✅ **StreamingServerUpstreamOp** - Frame reception operator
- ✅ **StreamingServerDownstreamOp** - Frame transmission operator
- ✅ **Resource sharing** - Multiple operators with shared resource
- ✅ **Python/C++ binding correctness** - Proper exposure to Python
- ✅ **Parameter handling** - Cross-language parameter passing
- ✅ **Memory management** - No leaks across boundaries

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

## Prerequisites

- Python 3.8+
- pytest 6.0+
- Holoscan SDK 3.5.0+
- StreamingServer Python bindings built

## Running Tests

### Option 1: Using pytest directly

```bash
# From this directory
pytest -v

# Run specific test
pytest -v -k test_resource_creation_basic

# Run specific test class
pytest -v test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding

# With coverage
pytest --cov=streaming_server_enhanced --cov-report=html
```

### Option 2: From operator root directory

```bash
# From operators/video_streaming/streaming_server_enhanced directory
python3 -m pytest python/tests/ -v
```

### Option 3: Via CMake/CTest

```bash
# From holohub root
./holohub build video_streaming --configure-args='-DBUILD_TESTING=ON'
./holohub test video_streaming --ctest-options="-R streaming_server.*pytest"
```

## Test Output Example

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 35 items

test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_resource_creation_basic PASSED [  2%]
test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_resource_name PASSED [  5%]
test_streaming_server_ops_bindings.py::TestStreamingServerResourceBinding::test_video_parameters[640-480-30] PASSED [  8%]
...
test_streaming_server_ops_bindings.py::TestStreamingServerIntegration::test_bidirectional_server_setup PASSED [ 97%]
test_streaming_server_ops_bindings.py::TestStreamingServerIntegration::test_operators_in_application_context PASSED [100%]

========================== 35 passed in 1.45s ==================================
```

## Fixtures

Defined in `conftest.py`:

**Core Fixtures:**
- `holoscan_modules` - Holoscan SDK imports
- `streaming_server_module` - StreamingServer Python module
- `streaming_server_classes` - All server operator classes
- `fragment` - Holoscan Fragment for testing

**Factory Fixtures:**
- `resource_factory` - Factory for creating StreamingServerResource
- `upstream_operator_factory` - Factory for creating upstream operators
- `downstream_operator_factory` - Factory for creating downstream operators

**Default Fixtures:**
- `default_resource` - Pre-configured resource with defaults

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

## Troubleshooting

### Import Errors

If you get `ImportError: cannot import name 'StreamingServerResource'`:

1. Ensure the Python bindings are built:
   ```bash
   ./holohub build video_streaming
   ```

2. Check PYTHONPATH includes the build directory:
   ```bash
   export PYTHONPATH=/path/to/holohub/build/python/lib:$PYTHONPATH
   ```

### Resource Creation Failures

If resource creation fails with `RuntimeError`:
- Ensure port is available (not in use)
- Check port is in valid range (1-65535)
- Verify no other server is using the same port

### Operator Creation Without Resource

If you get "Resource required for operator creation":
- Always pass a valid resource to operator factories
- Create resource first, then pass to operator factory

### Holoscan SDK Not Found

If you get `ModuleNotFoundError: No module named 'holoscan'`:

```bash
# Install Holoscan SDK or add to PYTHONPATH
export PYTHONPATH=/opt/nvidia/holoscan/python/lib:$PYTHONPATH
```

## Test Organization

```
python/tests/
├── conftest.py                                 # Pytest fixtures
├── test_streaming_server_ops_bindings.py       # Main test file
└── README.md                                   # This file
```

## Related Documentation

- [C++ Unit Tests](../../tests/README.md) - C++ operator unit tests
- [StreamingServer Documentation](../../README.md) - Operator documentation
- [Integration Tests](../../../../applications/video_streaming/TESTING.md) - End-to-end tests

## Contributing

When adding tests:
1. Follow existing test naming patterns (`test_<feature>_<aspect>`)
2. Use descriptive test names and docstrings
3. Use appropriate fixtures from `conftest.py`
4. Add parametrization for multiple test cases
5. Test both success and error cases
6. Update this README if adding new test categories
7. Ensure tests are isolated (no side effects)

## Test Marks

You can add custom markers:

```python
@pytest.mark.slow
def test_time_consuming():
    # Slow test
    pass

@pytest.mark.requires_gpu
def test_gpu_feature():
    # GPU-dependent test
    pass
```

Run specific markers:
```bash
pytest -m slow  # Run only slow tests
pytest -m "not slow"  # Skip slow tests
```

