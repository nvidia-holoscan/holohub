# Holoscan ROS2 Publisher/Subscriber Examples

## Overview
This application demonstrates basic communication between Holoscan and ROS2, containing both publisher and subscriber examples in a single combined structure. The examples show how to:
- Send simple string messages from Holoscan to ROS2 (Publisher)
- Receive messages from ROS2 in Holoscan operators (Subscriber)
- Bridge data between the two frameworks

Both C++ and Python implementations are included, with the publisher and subscriber components organized under shared `cpp/` and `python/` directories.

## File Structure
```
applications/holoscan_ros2/pubsub/
├── CMakeLists.txt          # Top-level build configuration
├── README.md               # This file
├── cpp/                    # C++ implementations
│   ├── CMakeLists.txt      # C++ build configuration
│   ├── metadata.json       # C++ application metadata
│   ├── talker.cpp          # Publisher implementation
│   └── listener.cpp        # Subscriber implementation
└── python/                 # Python implementations
    ├── CMakeLists.txt      # Python build configuration
    ├── metadata.json       # Python application metadata
    ├── talker.py           # Publisher implementation
    └── listener.py         # Subscriber implementation
```

> **Note:** This structure combines the previously separate `holoscan_ros2_simple_publisher` and `holoscan_ros2_simple_subscriber` applications into a single, more organized layout. All functionality remains the same, but both components are now built and managed together.

## Prerequisites
- **NVIDIA Holoscan SDK** v3.0 or later
- **ROS2 Jazzy** (all examples and Dockerfiles are tested with Jazzy; other distributions may work but are not tested)
- **Docker** (with NVIDIA Container Toolkit and a recent version supporting `--gpus all` and `--runtime=runc`)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)

> **Note:** When building and running any app, always use the Docker option:
> `--docker-opts "--runtime=runc --gpus all"`
> This ensures proper GPU and Vulkan support with recent Docker versions.

## Building the Application
Build the combined publisher/subscriber application:
```bash
./holohub build pubsub --docker-opts "--runtime=runc --gpus all"
```

This single command builds both the publisher and subscriber components for both C++ and Python.

## Application Modes

The pubsub application uses **modes** to handle the publisher and subscriber components. You can list available modes using:
```bash
./holohub modes pubsub --language cpp
./holohub modes pubsub --language python
```

Available modes:
- **publisher**: Sends simple string messages to ROS2 topic (default mode)
- **subscriber**: Receives messages from ROS2 topic

## Running C++ Publisher and Subscriber
Run the Publisher and the Subscriber components in different consoles. Use the `modes` feature to specify which component to run.

**Publisher component:**
```sh
./holohub run pubsub publisher --language cpp --docker-opts "--runtime=runc --gpus all"
# Or simply (since publisher is the default mode):
./holohub run pubsub --language cpp --docker-opts "--runtime=runc --gpus all"
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

**Subscriber component:**
```sh
./holohub run pubsub subscriber --language cpp --docker-opts "--runtime=runc --gpus all"
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
Run the Publisher and the Subscriber components in different consoles. Use the same `modes` as the C++ version.

**Publisher component:**
```sh
./holohub run pubsub publisher --language python --docker-opts "--runtime=runc --gpus all"
# Or simply (since publisher is the default mode):
./holohub run pubsub --language python --docker-opts "--runtime=runc --gpus all"
```

**Subscriber component:**
```sh
./holohub run pubsub subscriber --language python --docker-opts "--runtime=runc --gpus all"
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
