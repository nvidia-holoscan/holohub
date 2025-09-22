#!/bin/bash
# Script to run pytest tests in container

echo "🧪 Setting up pytest test environment..."

# Install pytest
pip install pytest numpy

# Navigate to test directory
cd /workspace/holohub/build/streaming_client_demo_enhanced_tests/applications/streaming_client_demo_enhanced_tests/operator_tests

# Temporarily move the conflicting conftest.py
if [ -f "/workspace/holohub/conftest.py" ]; then
    echo "📝 Temporarily moving global conftest.py to avoid conflicts..."
    mv /workspace/holohub/conftest.py /workspace/holohub/conftest.py.backup
fi

# Remove problematic pytest.ini
rm -f pytest.ini

# Set Python path
export PYTHONPATH="/workspace/holohub/build/streaming_client_demo_enhanced_tests/python/lib:$PYTHONPATH"

echo "🚀 Running pytest tests..."
echo "📁 Test directory: $(pwd)"
echo "🐍 Python path: $PYTHONPATH"
echo "📋 Available test files:"
ls -la test_*.py

# Run the tests
python3 -m pytest -v --tb=short test_streaming_client_op_bindings.py

# Store the result
TEST_RESULT=$?

# Restore the conftest.py if it was moved
if [ -f "/workspace/holohub/conftest.py.backup" ]; then
    echo "📝 Restoring global conftest.py..."
    mv /workspace/holohub/conftest.py.backup /workspace/holohub/conftest.py
fi

echo "✅ Test execution completed with exit code: $TEST_RESULT"
exit $TEST_RESULT
