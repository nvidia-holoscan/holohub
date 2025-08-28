# Holoscan ROS2 Application Examples

## What is ROS2?

ROS2 (Robot Operating System 2) is a set of software libraries and tools for building robot applications. Despite its name, ROS2 is not an operating system but rather a middleware framework that provides:

- **Communication infrastructure**: Publisher/subscriber messaging, services, and actions
- **Hardware abstraction**: Standardized interfaces for sensors, actuators, and other devices  
- **Package management**: Modular architecture with reusable components
- **Development tools**: Visualization, debugging, and simulation capabilities
- **Cross-platform support**: Linux, Windows, and macOS compatibility

ROS2 has become the de facto standard in robotics, with a vast ecosystem of packages for perception, navigation, manipulation, and more.

## Why Integrate Holoscan with ROS2?

### Complementary Strengths

**Holoscan** excels at:
- High-performance, low-latency streaming data processing
- GPU-accelerated computer vision and AI inference
- Real-time media processing (video, imaging, sensor data)
- Optimized memory management and zero-copy operations

**ROS2** excels at:
- Robot system integration and coordination
- Standardized interfaces and message formats
- Rich ecosystem of robotics packages
- Distributed system communication
- Hardware abstraction layers

### Use Cases for Holoscan-ROS2 Integration

1. **AI-Powered Robotics**
   - Process camera streams with Holoscan's GPU-accelerated vision pipeline
   - Send processed results (object detection, segmentation) to ROS2 navigation stack
   - Enable real-time decision making with sub-millisecond latency

2. **Medical Robotics**
   - Use Holoscan for real-time medical imaging and AI inference
   - Integrate with ROS2-based surgical robots or diagnostic systems
   - Maintain strict timing requirements for safety-critical applications

3. **Autonomous Vehicles**
   - Process high-resolution sensor data (cameras, LiDAR) in Holoscan
   - Interface with ROS2-based planning and control systems
   - Bridge between perception and decision-making components

4. **Industrial Automation**
   - Combine Holoscan's real-time vision processing with ROS2 robot controllers
   - Enable quality inspection, pick-and-place, and assembly operations
   - Maintain deterministic timing for production environments

5. **Research and Development**
   - Leverage existing ROS2 algorithms and tools
   - Add high-performance data processing capabilities
   - Prototype new robotics applications with state-of-the-art AI

## Application Examples

This directory contains example applications demonstrating Holoscan-ROS2 integration:

### [Publisher/Subscriber Examples](pubsub/)
Basic examples showing fundamental communication patterns between Holoscan and ROS2:
- Simple string message exchange
- Both C++ and Python implementations
- Bidirectional communication (Holoscan ↔ ROS2)

**Best for**: Learning the basics, understanding message flow, prototyping simple integrations

### [VB1940 Camera Examples](vb1940/)
Advanced examples using real camera hardware:
- High-resolution camera data processing
- GPU-accelerated image processing pipeline
- Real-time visualization and ROS2 topic publishing

**Best for**: Production-like scenarios, understanding performance characteristics, hardware integration

## Getting Started

1. **Choose your example**: Start with [pubsub examples](pubsub/) for basic concepts, or jump to [VB1940 examples](vb1940/) for advanced use cases
2. **Review prerequisites**: Each example has specific requirements listed in its README
3. **Build and run**: Follow the step-by-step instructions in each example's documentation
4. **Experiment**: Modify parameters, add custom processing, or integrate with your existing ROS2 system

## Architecture Overview

The Holoscan ROS2 bridge enables seamless data exchange through:

- **Message conversion**: Automatic translation between Holoscan tensors and ROS2 messages
- **Node management**: Integrated ROS2 node lifecycle within Holoscan applications
- **Threading model**: Non-blocking operations that don't interfere with real-time processing
- **Memory optimization**: Efficient data sharing between frameworks

For detailed technical information about the bridge implementation, see the [main bridge documentation](../../operators/holoscan_ros2/).

## Next Steps

- Explore the [simple examples](pubsub/) to understand basic integration patterns
- Try the [advanced examples](vb1940/) with real hardware
- Review the [bridge library documentation](../../operators/holoscan_ros2/) for API details
- Join the [Holoscan community](https://forums.developer.nvidia.com/) for support and discussions
