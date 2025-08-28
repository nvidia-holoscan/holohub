# Holoscan ROS2 Publisher/Subscriber Examples

## Overview
The Publisher/Subscriber examples demonstrate basic communication between Holoscan and ROS2. These examples show how to:
- Send simple string messages from Holoscan to ROS2
- Receive messages from ROS2 in Holoscan operators
- Bridge data between the two frameworks

## Prerequisites
- **NVIDIA Holoscan SDK** v3.0 or later
- **ROS2 Jazzy** (all examples and Dockerfiles are tested with Jazzy; other distributions may work but are not tested)
- **Docker** (with NVIDIA Container Toolkit and a recent version supporting `--gpus all` and `--runtime=runc`)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)

> **Note:** When building and running any app, always use the Docker option:
> `--docker-opts "--runtime=runc --gpus all"`
> This ensures proper GPU and Vulkan support with recent Docker versions.

## Building the Examples
First, build the examples:
```bash
./isaac-hub build holoscan_ros2_simple_publisher --docker-opts "--runtime=runc --gpus all"
./isaac-hub build holoscan_ros2_simple_subscriber --docker-opts "--runtime=runc --gpus all"
```

## Running C++ Publisher and Subscriber
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

## Running Python Publisher and Subscriber
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

## Additional Resources

### Documentation
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/Installation.html)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)

### Related Examples
- **Applications Overview**: `../` - Background on ROS2 and Holoscan integration
- **Camera Examples**: `../vb1940/` - Advanced camera integration with VB1940 hardware
- **Bridge Library**: `../../../operators/holoscan_ros2/` - Bridge implementation and headers

### Community and Support
- [Holoscan SDK GitHub](https://github.com/nvidia-holoscan/holoscan-sdk)
- [ROS2 Community](https://discourse.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
