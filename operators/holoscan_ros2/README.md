# Holoscan ROS2 Bridge Extension

## Overview
The Holoscan ROS2 Bridge extension provides interoperability between NVIDIA Holoscan and ROS2 (Robot Operating System 2) applications. It consists of:
- A C++ header-only library for seamless integration with ROS2 `rclcpp`-based applications
- A Python package for integration with ROS2 `rclpy`-based applications

Both implementations enable seamless data and message exchange between Holoscan SDK operators and ROS2 nodes. You can use either implementation depending on your preferred programming language. Example applications can be found under the `applications/holoscan_ros2` directory.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Publisher/Subscriber Examples](#publishersubscriber-examples)
  - [Running C++ Publisher and Subscriber](#running-c-publisher-and-subscriber)
  - [Running Python Publisher and Subscriber](#running-python-publisher-and-subscriber)
- [VB1940 (Eagle) Camera Examples](#vb1940-eagle-camera-examples)
  - [Troubleshooting VB1940 Examples](#troubleshooting-vb1940-examples)
- [Architecture](#architecture)
- [Additional Resources](#additional-resources)

## Prerequisites
- **NVIDIA Holoscan SDK** v3.0 or later
- **ROS2 Jazzy** (all examples and Dockerfiles are tested with Jazzy; other distributions may work but are not tested)
- **Docker** (with NVIDIA Container Toolkit and a recent version supporting `--gpus all` and `--runtime=runc`)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)
- **Git and SSH access** (for VB1940 examples requiring hololink repository)

For VB1940 camera examples, you'll also need:
- Access to NVIDIA's internal hololink repository
- VB1940 (Eagle) camera hardware
- Proper network configuration for camera communication

> **Note:** When building and running any app, always use the Docker option:
> `--docker-opts "--runtime=runc --gpus all"`
> This ensures proper GPU and Vulkan support with recent Docker versions.

## Publisher/Subscriber Examples
The Publisher/Subscriber examples demonstrate basic communication between Holoscan and ROS2. These examples are located under the `applications/holoscan_ros2/pubsub` folder and show how to:
- Send simple string messages from Holoscan to ROS2
- Receive messages from ROS2 in Holoscan operators
- Bridge data between the two frameworks

**First, build the examples:**
```bash
./isaac-hub build holoscan_ros2_simple_publisher --docker-opts "--runtime=runc --gpus all"
./isaac-hub build holoscan_ros2_simple_subscriber --docker-opts "--runtime=runc --gpus all"
```

### Running C++ Publisher and Subscriber
Run the Publisher and the Subscriber apps in different consoles.

**Publisher app:**
```sh
./isaac-hub run holoscan_ros2_simple_publisher --language cpp --docker-opts "--runtime=runc --gpus all"
```

Expected output:
```
Publishing: 'Hello, world! 12'
Publishing: 'Hello, world! 13'
Publishing: 'Hello, world! 14'
...
```

> **Troubleshooting:**
> If you see errors about missing Vulkan extensions or invalid Docker runtime, double-check that you are using `--docker-opts "--runtime=runc --gpus all"` for both build and run commands.

**Subscriber app:**
```sh
./isaac-hub run holoscan_ros2_simple_subscriber --language cpp --docker-opts "--runtime=runc --gpus all"
```

Expected output:
```
I heard: 'Hello, world! 12'
I heard: 'Hello, world! 13'
I heard: 'Hello, world! 14'
...
```

**Validation:**
- The publisher should output periodic "Publishing" logs with incrementing numbers
- The subscriber should output corresponding "I heard" logs with matching messages
- Messages should match between publisher and subscriber

### Running Python Publisher and Subscriber
Run the Publisher and the Subscriber apps in different consoles.

**Publisher app:**
```sh
./isaac-hub run holoscan_ros2_simple_publisher --language python --docker-opts "--runtime=runc --gpus all"
```

**Subscriber app:**
```sh
./isaac-hub run holoscan_ros2_simple_subscriber --language python --docker-opts "--runtime=runc --gpus all"
```

**Validation:**
- Similar to C++ examples, you should see matching publish/receive logs
- Python and C++ versions are interoperable (you can mix languages)

## VB1940 (Eagle) Camera Examples
The VB1940 (Eagle) Camera examples demonstrate advanced usage with real camera hardware. These examples show how to:
- Capture images from a VB1940 (Eagle) camera using Holoscan
- Process camera data through a complete pipeline
- Publish processed images to ROS2 topics for visualization or further processing
- Subscribe to and visualize camera streams

**Important:** These examples require access to NVIDIA's internal hololink repository and VB1940 camera hardware.

The `applications/holoscan_ros2/vb1940` folder contains examples for the VB1940 (Eagle) camera sensor:

### Network Setup
Before running VB1940 examples, ensure proper network configuration:
- Default Hololink board IP: `192.168.0.2`
- Ensure your host can reach the Hololink board
- Verify IBV (InfiniBand Verbs) device configuration if using custom hardware

1. **Publisher (`vb1940_publisher.cpp`)**
   - Captures images from a VB1940 (Eagle) camera using Holoscan
   - Processes the images through a pipeline:
     - CSI to Bayer conversion
     - Image processing
     - Bayer demosaicing
   - Publishes the processed images to ROS2 topic `vb1940/image`
   - Supports various camera modes and configurations

   Usage:
   ```bash
   $ ./isaac-hub run holoscan_ros2_vb1940_publisher [options] --docker-opts "--runtime=runc --gpus all"
   ```
   Options:
   - `--camera-mode`: VB1940 (Eagle) mode (default: 2560x1984 30FPS)
   - `--frame-limit`: Exit after publishing specified number of frames
   - `--hololink`: IP address of Hololink board (default: 192.168.0.2)
   - `--ibv-name`: IBV device to use
   - `--ibv-port`: Port number of IBV device (default: 1)

2. **Subscriber (`vb1940_subscriber.cpp`)**
   - Subscribes to the `vb1940/image` ROS2 topic
   - Receives and processes the images
   - Visualizes the images using Holoviz
   - Supports headless and fullscreen modes

   Usage:
   ```bash
   $ ./isaac-hub run holoscan_ros2_vb1940_subscriber [options] --docker-opts "--runtime=runc --gpus all"
   ```
   Options:
   - `--headless`: Run in headless mode
   - `--fullscreen`: Run in fullscreen mode

### Troubleshooting VB1940 Examples

#### SSH Access for Hololink Repository
The VB1940 examples require access to the internal hololink repository during Docker build. You need to ensure proper SSH setup:

1. **Start SSH Agent and Add Key:**
   ```bash
   # Start SSH agent if not running
   eval "$(ssh-agent -s)"

   # Add your SSH key (replace with your actual key path)
   ssh-add ~/.ssh/id_rsa

   # Verify the key is added
   ssh-add -l

   # Check your SSH_AUTH_SOCK value
   echo $SSH_AUTH_SOCK
   ```

2. **Build with SSH Agent Forwarding:**
   ```bash
   # Build with SSH forwarding (SSH_AUTH_SOCK is already set by ssh-agent)
   ./isaac-hub build holoscan_ros2_vb1940_publisher --build-args "--ssh default" --verbose --docker-opts "--runtime=runc --gpus all"
   ```

   **Note:** The `SSH_AUTH_SOCK` environment variable is automatically set when you start the SSH agent. You can verify it's set by running `echo $SSH_AUTH_SOCK` - it will show a path like `/tmp/ssh-XXXXXXhQl38c/agent.1662230`.

3. **Verify SSH Access:**
   ```bash
   # Test SSH connection to the repository server
   ssh -p 12051 git@gitlab-master.nvidia.com
   ```

If you encounter SSH-related build failures, ensure:
- Your SSH key has access to the hololink repository
- The SSH agent is running and your key is loaded
- The `SSH_AUTH_SOCK` environment variable is correctly set
- You're using the `--ssh default` build argument

#### Common Error Messages and Solutions

**Error: "Permission denied (publickey)"**
```bash
# Solution: Ensure your SSH key is added and has repository access
ssh-add -l  # Verify key is loaded
ssh -T -p 12051 git@gitlab-master.nvidia.com  # Test access
```

**Error: "Could not resolve hostname"**
```bash
# Solution: Check network connectivity and DNS resolution
ping gitlab-master.nvidia.com
```

**Error: "No such file or directory" during Docker build**
```bash
# Solution: Ensure you're using the --ssh flag
./isaac-hub build holoscan_ros2_vb1940_publisher --build-args "--ssh default" --docker-opts "--runtime=runc --gpus all"
```

## Architecture

The Holoscan ROS2 Bridge provides two main components:

### C++ Bridge (`holoscan::ros2`)
- **Header-only library** for easy integration
- **Bridge class** manages ROS2 node lifecycle and communication
- **Operator base class** simplifies creating ROS2-aware Holoscan operators
- **Type conversion** between Holoscan and ROS2 message types

### Python Bridge (`holoscan_ros2`)
- **Python package** for `rclpy`-based applications
- **Bridge resource** manages ROS2 node within Holoscan Python operators
- **Automatic message conversion** between Holoscan tensors and ROS2 messages
- **Threading support** for non-blocking ROS2 operations

Both implementations provide:
- Automatic ROS2 node management
- Message type conversion
- Publisher/Subscriber abstractions
- Integration with Holoscan's data flow architecture

## Additional Resources

### Documentation
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/Installation.html)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)

### Related Examples
- **Simple Examples**: `applications/holoscan_ros2/pubsub/` - Basic publisher/subscriber communication
- **Camera Examples**: `applications/holoscan_ros2/vb1940/` - Advanced camera integration
- **Operator Source**: `operators/holoscan_ros2/` - Bridge implementation and headers

### Community and Support
- [Holoscan SDK GitHub](https://github.com/nvidia-holoscan/holoscan-sdk)
- [ROS2 Community](https://discourse.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

### Development
- **C++ API Reference**: See `operators/holoscan_ros2/cpp/holoscan/ros2/` for headers
- **Python API Reference**: See `operators/holoscan_ros2/python/holoscan_ros2/` for implementation
- **Contributing**: Follow standard Holoscan SDK contribution guidelines
