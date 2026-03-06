# Building the ST2110 Source Operator

This document provides instructions for building the ST2110 Source operator on NVIDIA platforms.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU (any CUDA-capable GPU)
- Network interface with multicast support
- For Thor AGX: Built-in MGBE network interface

### Software Requirements
- Ubuntu 22.04 or later
- CMake 3.24 or later
- CUDA Toolkit 12.x or later
- Holoscan SDK 3.7.0 or later
- GCC 11+ with C++17 support

## Quick Start

```bash
# Build operator only
./holohub build st2110_source

# Build with Python bindings
./holohub build st2110_source --configure-args "-DHOLOHUB_BUILD_PYTHON=ON"
```

## Detailed Build Instructions

### Step 1: Verify Prerequisites

```bash
# Check CMake version (need 3.24+)
cmake --version

# Check CUDA installation
nvcc --version

# Check Holoscan SDK
ls /opt/nvidia/holoscan
```

### Step 2: Build the Operator

#### Option A: Using HoloHub build script (Recommended)

```bash
cd /path/to/holohub

# Build C++ operator
./holohub build st2110_source

# Build with Python bindings
./holohub build st2110_source --configure-args "-DHOLOHUB_BUILD_PYTHON=ON"
```

#### Option B: Direct CMake build (For development)

```bash
cd operators/st2110_source
mkdir -p build && cd build

cmake .. \
  -DCMAKE_PREFIX_PATH="/opt/nvidia/holoscan" \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

### Step 3: Verify the Build

```bash
# From holohub root (after Option A):
ls -lh build/st2110_source/operators/st2110_source/libst2110_source.so

# Or from operators/st2110_source/build (after Option B):
ls -lh libst2110_source.so
ldd libst2110_source.so | grep -E "(holoscan|cuda)"
```

## Build Configuration

### CMakeLists.txt Overview

The operator links against:
- `holoscan::core` - Holoscan SDK core library
- `CUDA::cuda_driver` - CUDA driver API for pinned memory

No external networking libraries are required - the operator uses standard Linux sockets.

### Environment Variables

```bash
# If CUDA is not in default path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# If Holoscan is not in default path
export CMAKE_PREFIX_PATH=/opt/nvidia/holoscan/lib:$CMAKE_PREFIX_PATH
```

## Common Build Issues

### Issue: "CMake version too old"

**Solution**: Install CMake 3.24+
```bash
# Download and install locally (auto-detects architecture)
ARCH=$(uname -m)
wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-${ARCH}.tar.gz
tar -xzf cmake-3.28.1-linux-${ARCH}.tar.gz -C $HOME/.local/ --strip-components=1
export PATH=$HOME/.local/bin:$PATH
```

### Issue: "CUDA not found"

**Solution**: Add CUDA to PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "holoscan package not found"

**Solution**: Verify Holoscan SDK installation
```bash
# Check installation
ls /opt/nvidia/holoscan

# Set CMake prefix path
export CMAKE_PREFIX_PATH=/opt/nvidia/holoscan/lib
```

## Platform-Specific Notes

### NVIDIA Thor AGX

Thor AGX uses the built-in MGBE network interface. No special network drivers are required.

```bash
# Verify MGBE interface
ip link show mgbe0_0
```

### x86_64 with ConnectX

Standard Linux network drivers work fine. No DPDK or special configuration needed.

## Testing the Build

After building, test with the included demo application:

```bash
# Run with holohub CLI
./holohub run st2110_demo

# Or run specific mode
./holohub run st2110_demo rgba   # RGBA output only
./holohub run st2110_demo nv12   # NV12 output only
```

Edit `applications/st2110_demo/st2110_demo_config.yaml` to match your network configuration.

## Next Steps

1. Configure your network interface for multicast (see README.md)
2. Set up your ST 2110 video source
3. Run the test application to verify reception
