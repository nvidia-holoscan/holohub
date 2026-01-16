#!/bin/bash
# Test 2: RGBA Only Configuration

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOLOHUB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Test 2: RGBA Only (NV12 Disabled)"
echo "========================================"
echo ""

docker run --rm --runtime=nvidia --network=host \
  --cap-add CAP_SYS_PTRACE --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${HOLOHUB_ROOT}:/workspace" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "/run/user/$(id -u):/run/user/$(id -u)" \
  -e "DISPLAY=$DISPLAY" \
  -e "XDG_RUNTIME_DIR=/run/user/$(id -u)" \
  -e PYTHONPATH=/opt/nvidia/holoscan/python/lib:/workspace/build/st2110_source/python/lib \
  -e LD_LIBRARY_PATH=/workspace/build/st2110_source/operators/st2110_source:/opt/nvidia/holoscan/lib \
  holohub:ngc-v3.7.0-cuda13 \
  bash -c "cd /workspace/applications/st2110_test && python3 st2110_test_app.py --config st2110_test_config_rgba_only.yaml"
