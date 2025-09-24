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


## 📁 Contents

### Core Test Files
- **`conftest.py`**: pytest configuration and fixtures for all tests
- **`test_streaming_client_op_bindings.py`**: Python binding validation tests
- **`test_operator_data_pipeline.py`**: **Data flow pipeline tests** 
- **`test_streaming_client_data_flow.py`**: **Comprehensive data flow tests** 
- **`test_utils.py`**: Utility functions, mock factories, and test helpers
- **`pytest.ini`**: pytest configuration file

### Mock Framework
- **`mock_holoscan_framework.py`**: **Complete mock Holoscan framework** (NEW)

### Configuration Files  
- **`unit_test_config.yaml`**: Minimal configuration for operator testing
- **`requirements.txt`**: Python dependencies for testing

### Build Integration
- **`CMakeLists.txt`**: CMake integration for pytest execution via CTest

## 🚀 How to Run Tests

### ✅ **Method 1: HoloHub Test Command (Easiest)**

```bash
# Run ALL tests (both C++ and Python) using holohub test:
./holohub test streaming_client_enhanced
```

This will automatically:
- Build the operator with BUILD_TESTING=ON
- Run both C++ (GTest) and Python (pytest) tests  
- Integrate with CTest for proper test discovery
- Show unified test results


## 🔄 **Data Flow Testing (NEW)**

The enhanced test suite now includes **comprehensive data flow testing** using mock frame data:

### **What's New:**
- **Mock TensorMap injection**: Test the `compute()` method with realistic BGR video data
- **Frame validation testing**: Verify `min_non_zero_bytes` validation logic  
- **BGR→BGRA conversion**: Test color format conversion with mock data
- **Complete pipeline testing**: End-to-end data flow from input to streaming client
- **Error scenario testing**: Malformed data, empty frames, client not ready

### **Mock Data Types:**
- **Gradient frames**: Smooth color transitions for visual validation
- **Solid color frames**: Uniform colors for specific testing scenarios  
- **Checkerboard patterns**: High-contrast patterns for edge detection
- **Noise frames**: Random patterns for stress testing
- **Empty frames**: All-zero data for validation threshold testing
- **Minimal content**: Frames with just enough data to test edge cases

### **Key Test Classes:**
- **`TestOperatorDataPipeline`**: Focused pipeline tests with realistic scenarios
- **`TestStreamingClientOpDataFlow`**: Comprehensive data flow validation
- **`TestDataValidationLogic`**: Frame validation and conversion logic
- **`TestStreamingClientOpErrorHandling`**: Error scenarios and edge cases

## 🏗️ Test Structure

### Unit Test #1: Python Help Test
- **Purpose**: Validate that the Python streaming client demo application provides proper help documentation
- **What it Tests**: Command-line help output, argument parsing, and usage information display
- **Expected Outcome**: Application displays help message with all available options (server_ip, signaling_port, width, height, fps)
- **How to Build & Execute the Test**: 
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

### Unit Test #2: Python Import Test
- **Purpose**: Verify that the streaming client demo Python module can be imported successfully
- **What it Tests**: Python module structure, import paths, and basic module functionality
- **Expected Outcome**: Module imports without errors and prints "✅ Import successful"
- **How to Build & Execute the Test**:
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

### Unit Test #3: Python Bindings Test
- **Purpose**: Validate that the StreamingClientOp C++ operator is accessible through Python bindings
- **What it Tests**: pybind11 binding functionality, C++ to Python interface
- **Expected Outcome**: StreamingClientOp class can be imported from holohub.streaming_client module
- **How to Build & Execute the Test**:
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

### Unit Test #4: Python Instantiation Test
- **Purpose**: Verify that the streaming client application can be instantiated with default parameters
- **What it Tests**: Object creation, parameter initialization, and basic application setup
- **Expected Outcome**: Application object created successfully with parameters: 640x480@30fps, server=127.0.0.1:48010
- **How to Build & Execute the Test**:
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

### Unit Test #5: Python Syntax Test
- **Purpose**: Ensure that the streaming client demo Python code has valid syntax
- **What it Tests**: Python syntax validation, compilation without execution
- **Expected Outcome**: Python module compiles successfully without syntax errors
- **How to Build & Execute the Test**:
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

### Unit Test #6: Python Parameters Test
- **Purpose**: Validate that the streaming client application accepts and handles custom parameters correctly
- **What it Tests**: Parameter passing, custom configuration, network and video parameter validation
- **Expected Outcome**: Application accepts custom parameters (1280x720@60fps, server=192.168.1.100:8080)
- **How to Build & Execute the Test**:
  ```bash
  ./holohub test streaming_client_demo_enhanced --verbose
  ```

## 🧩 Fixtures and Utilities

### Key Fixtures (from `conftest.py`)
- **`streaming_client_op_class`**: The imported `StreamingClientOp` class
- **`operator_factory`**: Factory function for creating operator instances
- **`mock_fragment`**: Mock Holoscan Fragment for testing
- **`mock_frame_data`**: Synthetic frame data generator
- **`default_operator_params`**: Standard parameter sets
- **`mock_holoscan_framework`**: **Complete mock Holoscan environment** (NEW)
- **`bgr_test_data`**: **Pre-generated BGR test frames** (NEW)

### Test Utilities (from `test_utils.py`)
- **`FrameDataGenerator`**: Create test video frames (gradients, checkerboards, noise)
- **`ParameterValidator`**: Validate video and network parameters
- **`MockStreamingClientFactory`**: Create mock clients for testing
- **`TestDataSets`**: Predefined test data for parametrized tests

### Mock Framework (from `mock_holoscan_framework.py`) ⭐ **NEW**
- **`MockTensor`**: Simulates Holoscan tensors with real numpy data
- **`MockTensorMap`**: Container for multiple tensors with dictionary-like access
- **`MockInputContext`**: Simulates operator input with configurable data
- **`MockOutputContext`**: Captures operator output for verification
- **`MockHoloscanFramework`**: Complete framework mock for end-to-end testing
- **`create_test_bgr_frame()`**: Generate various BGR test patterns
- **`create_test_scenario()`**: Pre-configured test scenarios
- **`validate_bgr_to_bgra_conversion()`**: Verify color format conversion

## 🎯 Test Categories (pytest markers)

```bash
@pytest.mark.unit           # Unit tests (isolated components)
@pytest.mark.integration    # Integration tests (component interactions)  
@pytest.mark.hardware       # Hardware-dependent tests (skip with --skip-hardware-tests)
@pytest.mark.slow           # Slow tests (> 5 seconds)
@pytest.mark.parametrized   # Parametrized tests (multiple scenarios)
```

## 🎯 **Expected Test Results**

**Application Tests (via HoloHub):**
```bash
Test project /workspace/holohub/build-streaming_client_demo_enhanced
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end

test 1
    Start 1: streaming_client_demo_enhanced_python_help_test
1/6 Test #1: streaming_client_demo_enhanced_python_help_test ............   Passed    0.14 sec

test 2
    Start 2: streaming_client_demo_enhanced_python_import_test
2/6 Test #2: streaming_client_demo_enhanced_python_import_test ..........   Passed    0.13 sec

test 3
    Start 3: streaming_client_demo_enhanced_python_bindings_test
3/6 Test #3: streaming_client_demo_enhanced_python_bindings_test ........   Passed    0.13 sec

test 4
    Start 4: streaming_client_demo_enhanced_python_instantiation_test
4/6 Test #4: streaming_client_demo_enhanced_python_instantiation_test ...   Passed    0.13 sec

test 5
    Start 5: streaming_client_demo_enhanced_python_syntax_test
5/6 Test #5: streaming_client_demo_enhanced_python_syntax_test ..........   Passed    0.03 sec

test 6
    Start 6: streaming_client_demo_enhanced_python_parameters_test
6/6 Test #6: streaming_client_demo_enhanced_python_parameters_test ......   Passed    0.13 sec

100% tests passed, 0 tests failed out of 6

Total Test time (real) =   0.69 sec
```

**Operator Python Tests (pytest - when available):**
```bash
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /workspace/holohub
configfile: pyproject.toml
collected X items

test_streaming_client_op_bindings.py ...............................     [100%]

============================== X passed in 0.19s ==============================
```

**Operator C++ Tests (GTest - when available):**
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
