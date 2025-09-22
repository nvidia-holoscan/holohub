# 🧪 How to Run pytest Tests for streaming_client_enhanced_test

## ✅ **Method 1: Simple Docker Command (Recommended)**

```bash
# Run this single command to execute all pytest tests:
docker run --rm -it \
  --net host \
  -v $PWD:/workspace/holohub \
  -w /workspace/holohub \
  holohub:streaming_client_demo_enhanced_tests \
  bash -c '
  pip install pytest numpy && \
  cd /workspace/holohub/build/streaming_client_demo_enhanced_tests/applications/streaming_client_demo_enhanced_tests/operator_tests && \
  rm -f pytest.ini && \
  PYTHONPATH="/workspace/holohub/build/streaming_client_demo_enhanced_tests/python/lib" \
  python3 -m pytest -v --tb=short test_streaming_client_op_bindings.py'
```

## ✅ **Method 2: Using the Helper Script**

```bash
# Run the provided helper script:
python3 run_tests_in_container.py
```

## ✅ **Method 3: Manual Container Entry**

```bash
# 1. Enter the container
docker run --rm -it \
  --net host \
  -v $PWD:/workspace/holohub \
  -w /workspace/holohub \
  holohub:streaming_client_demo_enhanced_tests \
  bash

# 2. Inside container, install pytest
pip install pytest numpy

# 3. Set up environment and run tests
cd /workspace/holohub/build/streaming_client_demo_enhanced_tests/applications/streaming_client_demo_enhanced_tests/operator_tests
export PYTHONPATH="/workspace/holohub/build/streaming_client_demo_enhanced_tests/python/lib"
rm -f pytest.ini  # Remove problematic config
python3 -m pytest -v --tb=short test_streaming_client_op_bindings.py
```

## 🎯 **What You'll See When Tests Run**

```bash
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /workspace/holohub
configfile: pyproject.toml
collected 31 items

test_streaming_client_op_bindings.py ...............................     [100%]

============================== 31 passed in 0.19s ==============================
```

**🎉 ALL 31 TESTS PASSING!**

## 🔧 **Why This Approach Works**

- ✅ **Holoscan SDK Available**: Docker container has all dependencies
- ✅ **CUDA Libraries**: GPU libraries available in container
- ✅ **Python Bindings**: Compiled operator bindings accessible
- ✅ **Clean Environment**: No conflicting conftest.py files
- ✅ **Proper PYTHONPATH**: Build directory included in Python path

## 📋 **Test Categories Available**

Pytests include:

- **Basic Operator Tests**: Creation, initialization
- **Parameter Tests**: Video dimensions, network settings
- **Parametrized Tests**: Multiple configuration combinations
- **Error Handling**: Invalid parameter validation
- **Mock Tests**: Behavior with mocked dependencies
- **Type Validation**: Parameter type checking

## 🚨 **Troubleshooting**

If tests fail with import errors:
1. Make sure the container was built: `./holohub build streaming_client_demo_enhanced_tests`
2. Check Python bindings exist: `ls build/streaming_client_demo_enhanced_tests/python/lib/holohub/`
3. Verify operator compiled: `ls build/streaming_client_demo_enhanced_tests/operators/`


