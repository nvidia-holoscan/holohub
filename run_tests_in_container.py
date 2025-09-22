#!/usr/bin/env python3
"""
Script to run pytest tests inside the HoloHub Docker container.

This script provides the correct way to run the operator pytest tests
where all dependencies (Holoscan SDK, CUDA) are available.
"""

import subprocess
import sys

def main():
    print("🧪 Running pytest tests for streaming_client_enhanced_test")
    print("=" * 60)
    
    # Docker command to run tests
    docker_cmd = [
        "docker", "run", "--rm", "-it",
        "--net", "host",
        "-v", "/home/cdinea/Downloads/enhancedapp_holohub/holohub:/workspace/holohub",
        "-w", "/workspace/holohub",
        "-e", "PYTHONPATH=/workspace/holohub/build/streaming_client_demo_enhanced_tests/python/lib",
        "holohub:streaming_client_demo_enhanced_tests",
        "bash", "-c",
        """
        # Install pytest in container
        pip install pytest numpy
        
        # Navigate to test directory
        cd /workspace/holohub/build/streaming_client_demo_enhanced_tests/applications/streaming_client_demo_enhanced_tests/operator_tests
        
        # Remove problematic pytest.ini if it exists
        rm -f pytest.ini
        
        # Run tests
        python3 -m pytest -v --tb=short --build-dir="../../../" test_streaming_client_op_bindings.py
        """
    ]
    
    print("🚀 Running command in Docker container...")
    print("📝 This will:")
    print("   1. Install pytest in the container")
    print("   2. Set up proper Python paths")
    print("   3. Run the operator pytest tests")
    print("   4. Show detailed test results")
    print()
    
    try:
        result = subprocess.run(docker_cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running Docker command: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
