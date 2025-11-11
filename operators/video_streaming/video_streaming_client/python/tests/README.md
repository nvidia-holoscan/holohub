# Python Unit Tests for StreamingClientOp Bindings

This directory contains Python unit tests for the `StreamingClientOp` Python bindings (pybind11).

## Overview

The pytest tests verify the Python bindings of the StreamingClientOp operator, focusing on:

- ✅ **Python/C++ binding correctness** - Proper exposure of C++ classes to Python
- ✅ **Parameter handling** - Correct parameter passing across language boundaries  
- ✅ **Method availability** - All required methods accessible from Python
- ✅ **Inheritance** - Proper inheritance from Holoscan base classes
- ✅ **Memory management** - No leaks across Python/C++ boundary
- ✅ **Error handling** - Graceful handling of errors in Python context

## Test Coverage

### TestStreamingClientOpBinding

**Basic Tests:**
- `test_operator_creation_basic` - Basic operator instantiation
- `test_operator_name` - Name property validation
- `test_operator_inheritance` - Inheritance from holoscan.Operator
- `test_method_availability` - Required methods are exposed

**Parametrized Tests:**
- `test_video_parameters` - Various resolutions and frame rates (VGA, HD, Full HD, 4K)
- `test_network_parameters` - Different IPs and ports
- `test_streaming_mode_parameters` - All send/receive combinations
- `test_frame_validation_parameter` - min_non_zero_bytes thresholds

**Advanced Tests:**
- `test_setup_method` - setup() method functionality
- `test_memory_management` - Multiple operators, garbage collection
- `test_operator_reuse` - Multiple instances with same parameters
- `test_string_parameter_handling` - String handling across boundaries
- `test_edge_case_resolutions` - Min/max resolutions
- `test_edge_case_fps` - Min/max frame rates
- `test_edge_case_ports` - Min/max port numbers
- `test_multiple_instances_isolation` - Instance isolation

### TestStreamingClientOpIntegration

- `test_operator_in_application_context` - Works within Application
- `test_operator_in_fragment` - Works within Fragment

### TestStreamingClientOpWithMockData

**Mock Data Tests:**
- `test_operator_with_mock_frame_cupy` - CuPy array handling
- `test_operator_with_mock_frame_numpy` - NumPy array handling
- `test_operator_with_various_resolutions` - Multiple resolutions
- `test_operator_with_grayscale_frame` - Grayscale format
- `test_operator_with_float_frame` - Float32 data type
- `test_operator_with_seeded_frame` - Deterministic random data

### TestStreamingClientOpCompute

**Compute Method Tests:**
- `test_compute_method_exists` - Method accessibility
- `test_client_compute_with_mock_frame` - Basic compute functionality
- `test_client_compute_with_various_resolutions` - Multiple resolutions
- `test_client_compute_with_float_frames` - Float32 frames
- `test_client_compute_method_signature` - Method signature validation

## ⚠️ CUDA Version Compatibility

> **Important:** These tests require **CUDA 12**. If you're using CUDA 13, you must specify `--cuda 12` when building and running tests.

**For CUDA 13 systems:**
```bash
# Building
./holohub build video_streaming --cuda 12 --configure-args='-DBUILD_TESTING=ON'

# Running tests
./holohub test video_streaming --cuda 12 --ctest-options="-R video_streaming_client_pytest -VV"
```

The video streaming client operator depends on libraries built against CUDA 12 runtime. Using CUDA 13 without the `--cuda 12` flag will cause test failures.

## Prerequisites

- Python 3.8+
- pytest 6.0+
- Holoscan SDK 3.5.0+
- StreamingClientOp Python bindings built
- **CUDA 12** (or use `--cuda 12` flag on CUDA 13 systems)

## Running Tests

### Option 1: Via CMake/CTest (Recommended)

```bash
# From holohub root
./holohub build video_streaming --cuda 12 --configure-args='-DBUILD_TESTING=ON'
./holohub test video_streaming --cuda 12 --ctest-options="-R video_streaming_client_pytest -VV"
```

### Option 2: Direct pytest Inside Container

For interactive testing and debugging, you can run pytest directly inside the holohub container:

```bash
# Launch interactive container with bash
./holohub run-container video_streaming --cuda 12 --docker-opts='--entrypoint=bash'

# Inside container, run all client tests
pytest operators/video_streaming/video_streaming_client/python/tests/ -v

# Inside container, run specific test class
pytest operators/video_streaming/video_streaming_client/python/tests/test_streaming_client_op_bindings.py::TestStreamingClientOpBinding -v

# Inside container, run specific test method
pytest operators/video_streaming/video_streaming_client/python/tests/test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_operator_creation_basic -v

# Inside container, run with coverage
pytest operators/video_streaming/video_streaming_client/python/tests/ -v \
  --cov=video_streaming_client \
  --cov-report=html
```

> **Note:** Running pytest directly requires being inside the holohub container where CUDA libraries and Python dependencies are properly configured. Direct pytest execution outside the container will fail with CUDA library errors.

## Test Output Example

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-7.4.0
collected 33 items

test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_operator_creation_basic PASSED [  4%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_operator_name PASSED [  8%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_video_parameters[640-480-30] PASSED [ 12%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_video_parameters[1280-720-60] PASSED [ 16%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_video_parameters[1920-1080-30] PASSED [ 20%]
...
test_streaming_client_op_bindings.py::TestStreamingClientOpWithMockData::test_operator_with_mock_frame_cupy PASSED [ 90%]
test_streaming_client_op_bindings.py::TestStreamingClientOpCompute::test_compute_method_exists PASSED [ 95%]
...
========================== 33 passed in 2.5s ==================================
```

## Fixtures

Defined in `conftest.py`:

**Module Fixtures:**
- `streaming_client_module` - VideoStreamingClient Python module
- `streaming_client_op_class` - VideoStreamingClientOp class

**Factory Fixtures:**
- `operator_factory` - Factory for creating operators with parameters
- `default_operator` - Pre-configured operator with defaults

**Common Fixtures** (copied from root conftest.py):
- `app` - Holoscan Application instance
- `fragment` - Holoscan Fragment for testing
- `mock_image` - Factory for creating mock image tensors
- `op_input_factory` - Factory for mock operator inputs
- `op_output` - Mock operator output
- `execution_context` - Mock execution context

> **Note:** Common fixtures are duplicated locally because `add_python_tests()` uses `--confcutdir` for test isolation, preventing access to the root conftest.py.

## Writing New Tests

To add new tests:

1. Add test function in `test_streaming_client_op_bindings.py`:
```python
def test_my_new_feature(self, operator_factory):
    """Test description."""
    op = operator_factory(custom_param=value)
    assert op is not None
    # Add assertions
```

2. Use fixtures for setup:
```python
def test_with_custom_setup(self, fragment, streaming_client_op_class):
    op = streaming_client_op_class(fragment, name="test", ...)
    # Test logic
```

3. Use parametrize for multiple cases:
```python
@pytest.mark.parametrize("param1,param2", [
    (value1, value2),
    (value3, value4),
])
def test_parametrized(self, operator_factory, param1, param2):
    # Test logic
```

## Troubleshooting

### Import Errors

If you get `ImportError: cannot import name 'StreamingClientOp'`:

1. Ensure the Python bindings are built:
   ```bash
   ./holohub build video_streaming
   ```

2. Check PYTHONPATH includes the build directory:
   ```bash
   export PYTHONPATH=/path/to/holohub/build/python/lib:$PYTHONPATH
   ```

### Holoscan SDK Not Found

If you get `ModuleNotFoundError: No module named 'holoscan'`:

```bash
# Install Holoscan SDK or add to PYTHONPATH
export PYTHONPATH=/opt/nvidia/holoscan/python/lib:$PYTHONPATH
```

### Test Skipping

Some tests may be skipped if:
- Holoscan SDK is not available
- Python bindings are not built
- Required dependencies are missing

This is expected behavior - tests will skip gracefully.

## Related Documentation

- [C++ Unit Tests](../../tests/README.md) - C++ operator unit tests
- [StreamingClientOp Documentation](../../README.md) - Operator documentation
- [Integration Tests](../../../../../applications/video_streaming/TESTING.md) - End-to-end tests

## Contributing

When adding tests:
1. Follow existing test naming patterns
2. Use descriptive test names
3. Add docstrings explaining what is tested
4. Use appropriate fixtures
5. Add parametrization for multiple test cases
6. Update this README if adding new test categories

