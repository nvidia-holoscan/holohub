# Holoscan ROS2 Bridge Extension

## Overview
The Holoscan ROS2 Bridge extension provides interoperability between NVIDIA Holoscan and ROS2 (Robot Operating System 2) applications. It consists of:
- A C++ header-only library for seamless integration with ROS2 `rclcpp`-based applications
- A Python package for integration with ROS2 `rclpy`-based applications

Both implementations enable seamless data and message exchange between Holoscan SDK operators and ROS2 nodes. You can use either implementation depending on your preferred programming language. Example applications can be found under the `applications/holoscan_ros2` directory.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Additional Resources](#additional-resources)

## Prerequisites
- **NVIDIA Holoscan SDK** v3.0 or later
- **ROS2 Jazzy** (tested with Jazzy; other distributions may work but are not tested)

For development:
- **C++ compiler** with C++17 support (for C++ bridge)
- **Python 3.8+** (for Python bridge)
- **CMake** (for building C++ applications)

For examples and testing:
- **Docker** (with NVIDIA Container Toolkit)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)

> **Note:** Example applications are available in the `applications/holoscan_ros2/` directory. Refer to their respective README files for specific requirements and instructions.

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
- **Simple Examples**: [`applications/holoscan_ros2/pubsub/`](../../applications/holoscan_ros2/pubsub/) - Basic publisher/subscriber communication
- **Camera Examples**: [`applications/holoscan_ros2/vb1940/`](../../applications/holoscan_ros2/vb1940/) - Advanced camera integration with VB1940 hardware
- **Operator Source**: `operators/holoscan_ros2/` - Bridge implementation and headers

### Community and Support
- [Holoscan SDK GitHub](https://github.com/nvidia-holoscan/holoscan-sdk)
- [ROS2 Community](https://discourse.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

### Development
- **C++ API Reference**: See `operators/holoscan_ros2/cpp/holoscan/ros2/` for headers
- **Python API Reference**: See `operators/holoscan_ros2/python/holoscan_ros2/` for implementation
- **Contributing**: Follow standard Holoscan SDK contribution guidelines
