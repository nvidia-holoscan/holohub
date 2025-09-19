# Unit Testing for Streaming Client Enhanced Operator

This directory contains **unit tests** for the `StreamingClientOp` operator, focusing on testing the **Python bindings (pybind11)** and core operator functionality in isolation.

## 🎯 Purpose

- **Python Binding Validation**: Test pybind11 bindings for the `StreamingClientOp` C++ class
- **Parameter Validation**: Verify parameter passing across Python/C++ boundary
- **Isolate Components**: Test operator logic without requiring external dependencies
- **Fast Feedback**: Provide quick validation during development
- **Regression Prevention**: Catch bugs in Python bindings and operator internals

## 🧪 Test Framework: **pytest**

These unit tests use **pytest** for Python testing, following the same pattern as other HoloHub operators like `adv_networking_bench`.

### Why pytest?
- **Powerful fixtures**: Reusable test components and mock objects
- **Parametrized testing**: Test multiple parameter combinations efficiently  
- **Clear assertions**: Descriptive test failures
- **Extensive ecosystem**: Rich plugin ecosystem for advanced testing needs

## 📁 Contents

### Core Test Files
- **`conftest.py`**: pytest configuration and fixtures for all tests
- **`test_streaming_client_op_bindings.py`**: Main unit tests for Python bindings
- **`test_utils.py`**: Utility functions, mock factories, and test helpers
- **`pytest.ini`**: pytest configuration file

### Configuration Files  
- **`unit_test_config.yaml`**: Minimal configuration for operator testing

### Build Integration
- **`CMakeLists.txt`**: CMake integration for pytest execution via CTest

## 🚀 Usage

### Option 1: Direct pytest Execution (Recommended)
```bash
# From the testing directory
cd operators/streaming_client_enhanced/testing

# Run all tests
pytest -v

# Run specific test categories
pytest -v -m unit                    # Unit tests only
pytest -v -m "not hardware"          # Skip hardware tests
pytest -v -m "parametrized"          # Parametrized tests only

# Run with custom build directory
pytest -v --build-dir=/path/to/build

# Skip hardware-dependent tests
pytest -v --skip-hardware-tests
```

### Option 2: CTest Integration
```bash
# Build with testing enabled
cmake -B build -DBUILD_TESTING=ON
cmake --build build

# Run pytest through CTest
cd build
ctest -R streaming_client_enhanced_python_unit_tests -V
```

## 🏗️ Test Structure

### 1. **Binding Tests** (`TestStreamingClientOpBinding`)
- ✅ **Operator Creation**: Basic pybind11 binding functionality
- ✅ **Parameter Validation**: Type checking and value validation
- ✅ **Method Availability**: Ensure C++ methods are accessible from Python
- ✅ **Memory Management**: Python/C++ boundary memory handling
- ✅ **Error Handling**: Exception propagation from C++ to Python

### 2. **Integration Tests** (`TestStreamingClientOpIntegration`) 
- ✅ **Application Context**: Operator within Holoscan Application
- ✅ **Parameter Persistence**: Parameter storage and retrieval
- ✅ **Context Integration**: Fragment and Application integration

### 3. **Error Handling Tests** (`TestStreamingClientOpErrorHandling`)
- ✅ **Exception Propagation**: C++ exceptions → Python exceptions
- ✅ **Invalid Parameters**: Proper handling of bad inputs
- ✅ **Type Safety**: Type validation across language boundary

### 4. **Performance Tests** (`TestStreamingClientOpPerformance`)
- ✅ **Creation Performance**: Operator instantiation speed
- ✅ **Memory Usage**: Memory leak detection
- ✅ **Resource Management**: Cleanup and resource handling

## 🧩 Fixtures and Utilities

### Key Fixtures (from `conftest.py`)
- **`streaming_client_op_class`**: The imported `StreamingClientOp` class
- **`operator_factory`**: Factory function for creating operator instances
- **`mock_fragment`**: Mock Holoscan Fragment for testing
- **`mock_frame_data`**: Synthetic frame data generator
- **`default_operator_params`**: Standard parameter sets

### Test Utilities (from `test_utils.py`)
- **`FrameDataGenerator`**: Create test video frames (gradients, checkerboards, noise)
- **`ParameterValidator`**: Validate video and network parameters
- **`MockStreamingClientFactory`**: Create mock clients for testing
- **`TestDataSets`**: Predefined test data for parametrized tests

## 🎯 Test Categories (pytest markers)

```bash
@pytest.mark.unit           # Unit tests (isolated components)
@pytest.mark.integration    # Integration tests (component interactions)  
@pytest.mark.hardware       # Hardware-dependent tests (skip with --skip-hardware-tests)
@pytest.mark.slow           # Slow tests (> 5 seconds)
@pytest.mark.parametrized   # Parametrized tests (multiple scenarios)
```

## 📊 Example Test Execution

```bash
$ pytest -v

=================== test session starts ===================
collected 25 items

test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_operator_creation_basic PASSED                    [  4%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_video_parameters[640-480-30] PASSED               [  8%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_video_parameters[1280-720-60] PASSED              [ 12%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_network_parameters[127.0.0.1-48010] PASSED        [ 16%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_parameter_type_validation PASSED                  [ 20%]
test_streaming_client_op_bindings.py::TestStreamingClientOpBinding::test_invalid_parameters_handling PASSED                [ 24%]
...
=================== 25 passed in 2.45s ===================
```

## 🔧 Dependencies

### Required for Tests
- **pytest**: Core testing framework
- **numpy**: Frame data generation and manipulation
- **Holoscan SDK**: Core framework (Fragment, Application, Operator)
- **StreamingClientOp Python bindings**: The operator being tested

### Optional for Advanced Testing
- **pytest-timeout**: Test timeout management
- **pytest-mock**: Enhanced mocking capabilities
- **pytest-cov**: Code coverage analysis

## 🚦 Running Specific Test Types

```bash
# Test only Python binding functionality
pytest -v -k "binding"

# Test parameter validation
pytest -v -k "parameter"

# Test error handling
pytest -v -k "error"

# Test performance (slow tests)
pytest -v -m slow

# Test with different video resolutions
pytest -v -k "video_parameters"
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the build directory is in Python path
   ```bash
   export PYTHONPATH="/path/to/build/python/lib:$PYTHONPATH"
   ```

2. **Missing Operator**: Ensure StreamingClientOp is built with Python bindings
   ```bash
   cmake -DBUILD_PYTHON=ON -DBUILD_TESTING=ON
   ```

3. **Hardware Tests Failing**: Skip hardware tests if no hardware available
   ```bash
   pytest --skip-hardware-tests
   ```

This testing approach provides comprehensive validation of the StreamingClientOp Python bindings while maintaining fast execution and clear organization! 🚀