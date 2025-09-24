# Testing for Streaming Client Enhanced Operator

This directory contains **both C++ and Python unit tests** for the `StreamingClientOp` operator:
- **C++ Unit Tests**: GTest-based tests for core operator functionality
- **Python Unit Tests**: pytest-based tests for Python bindings (pybind11)

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

## 🚀 How to Run Tests

### ✅ **Method 1: HoloHub Test Command (Easiest)**

```bash
# Run ALL tests (both C++ and Python) using holohub test:
./holohub test streaming_client_enhanced_test
```

This will automatically:
- Build the operator with BUILD_TESTING=ON
- Run both C++ (GTest) and Python (pytest) tests  
- Integrate with CTest for proper test discovery
- Show unified test results

### ✅ **Method 2: Manual pytest Execution**

```bash
# From the testing directory
cd operators/streaming_client_enhanced_test/testing

# Run all Python tests
pytest -v

# Run specific test categories
pytest -v -m unit                    # Unit tests only
pytest -v -m "not hardware"          # Skip hardware tests
pytest -v -m "parametrized"          # Parametrized tests only

# Run with custom build directory
pytest -v --build-dir=/path/to/build
```

### ✅ **Method 3: Docker Container Execution**

```bash
# Run pytest tests in Docker container:
docker run --rm -it \
  --net host \
  -v $PWD:/workspace/holohub \
  -w /workspace/holohub \
  holohub:streaming_client_demo_enhanced_tests \
  bash -c '
  pip install pytest numpy && \
  cd /workspace/holohub/build/streaming_client_demo_enhanced_tests/applications/streaming_client_demo_enhanced_tests/operator_tests && \
  PYTHONPATH="/workspace/holohub/build/streaming_client_demo_enhanced_tests/python/lib" \
  python3 -m pytest -v --tb=short test_streaming_client_op_bindings.py'
```

### ✅ **Method 4: CTest Integration**

```bash
# Build with testing enabled
cmake -B build -DBUILD_TESTING=ON
cmake --build build

# Run tests through CTest
cd build
ctest -R streaming_client_enhanced -V
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

## 🎯 **Expected Test Results**

**Python Tests (pytest):**
```bash
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /workspace/holohub
configfile: pyproject.toml
collected 31 items

test_streaming_client_op_bindings.py ...............................     [100%]

============================== 31 passed in 0.19s ==============================
```

**C++ Tests (GTest):**
```bash
[==========] Running X tests from Y test suites.
[----------] Global test environment set-up.
[----------] X tests from StreamingClientOpTest
[ RUN      ] StreamingClientOpTest.BasicInitialization
[       OK ] StreamingClientOpTest.BasicInitialization (X ms)
# ... more tests ...
[==========] X tests from Y test suites ran. (X ms total)
[  PASSED  ] X tests.
```

**🎉 ALL TESTS PASSING!**

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

## 🚨 **Troubleshooting**

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

4. **Container Issues**: Make sure the container was built
   ```bash
   ./holohub build streaming_client_demo_enhanced_tests
   ```

5. **Python Bindings Missing**: Check if bindings exist
   ```bash
   ls build/streaming_client_demo_enhanced_tests/python/lib/holohub/
   ```

6. **Operator Not Compiled**: Verify operator compiled successfully
   ```bash
   ls build/streaming_client_demo_enhanced_tests/operators/
   ```

### Why This Approach Works

- ✅ **Holoscan SDK Available**: Docker container has all dependencies
- ✅ **CUDA Libraries**: GPU libraries available in container
- ✅ **Python Bindings**: Compiled operator bindings accessible
- ✅ **Clean Environment**: No conflicting conftest.py files
- ✅ **Proper PYTHONPATH**: Build directory included in Python path

This comprehensive testing approach provides validation of both C++ core functionality and Python bindings while maintaining fast execution and clear organization! 🚀