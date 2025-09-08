# Holoscan ROS2 VB1940 (Eagle) Camera

## Overview
This unified VB1940 (Eagle) Camera application demonstrates advanced usage with real camera hardware, containing both publisher and subscriber components in a single structure. The application shows how to:
- Capture images from a VB1940 (Eagle) camera using Holoscan (Publisher mode)
- Process camera data through a complete pipeline
- Publish processed images to ROS2 topics for visualization or further processing
- Subscribe to and visualize camera streams (Subscriber mode)

The application uses **modes** to run either the publisher or subscriber component, with both C++ implementations organized under a unified `cpp/` directory.

**Important:** This application requires access to NVIDIA's internal hololink repository and VB1940 camera hardware.

## File Structure
```
applications/holoscan_ros2/vb1940/
├── CMakeLists.txt          # Top-level build configuration
├── README.md               # This file
└── cpp/                    # C++ implementation
    ├── CMakeLists.txt      # C++ build configuration
    ├── metadata.json       # Application metadata with modes
    ├── vb1940_publisher.cpp    # Publisher implementation
    └── vb1940_subscriber.cpp   # Subscriber implementation
```

*Note: This structure combines the previously separate `holoscan_ros2_vb1940_publisher` and `holoscan_ros2_vb1940_subscriber` applications.*

## Prerequisites
- **NVIDIA Holoscan SDK** v3.0 or later
- **ROS2 Jazzy** (all examples and Dockerfiles are tested with Jazzy; other distributions may work but are not tested)
- **Docker** (with NVIDIA Container Toolkit)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)
- **Git and SSH access** (for VB1940 examples requiring hololink repository)

For VB1940 camera examples, you'll also need:
- Access to NVIDIA's internal hololink repository
- VB1940 (Eagle) camera hardware
- Proper network configuration for camera communication

## Network Setup
Before running VB1940 examples, ensure proper network configuration:
- Default Hololink board IP: `192.168.0.2`
- Ensure your host can reach the Hololink board
- Verify IBV (InfiniBand Verbs) device configuration if using custom hardware

## Building the Application
Build the unified VB1940 camera application:
```bash
./holohub build vb1940 --build-args="--ssh default"
```

**Note:** The `--build-args="--ssh default"` is required for accessing the internal hololink repository during build.

## Application Modes

The vb1940 application uses **modes** to handle the publisher and subscriber components. You can list available modes using:
```bash
./holohub modes vb1940 --language cpp
```

Available modes:
- **publisher**: Captures and publishes VB1940 camera images (default mode)
- **subscriber**: Receives and visualizes camera images

## Running the Application

### Publisher Mode (`vb1940_publisher.cpp`)
Captures images from a VB1940 (Eagle) camera using Holoscan and publishes to ROS2:
- Processes images through a complete pipeline:
  - CSI to Bayer conversion
  - Image processing
  - Bayer demosaicing
- Publishes processed images to ROS2 topic `vb1940/image`
- Supports various camera modes and configurations

**Usage:**
```bash
# Run publisher (default mode)
./holohub run vb1940 [options]
# Or explicitly specify publisher mode:
./holohub run vb1940 publisher [options]
```

**Publisher Options:**
- `--camera-mode`: VB1940 (Eagle) mode (default: 2560x1984 30FPS)
- `--frame-limit`: Exit after publishing specified number of frames
- `--hololink`: IP address of Hololink board (default: 192.168.0.2)
- `--ibv-name`: IBV device to use
- `--ibv-port`: Port number of IBV device (default: 1)

### Subscriber Mode (`vb1940_subscriber.cpp`)
Subscribes to camera images from ROS2 and visualizes them:
- Subscribes to the `vb1940/image` ROS2 topic
- Receives and processes the images
- Visualizes images using Holoviz
- Supports headless and fullscreen modes

**Usage:**
```bash
./holohub run vb1940 subscriber [options]
```

**Subscriber Options:**
- `--headless`: Run in headless mode
- `--fullscreen`: Run in fullscreen mode


## Additional Resources

### Documentation
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/Installation.html)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)

### Hardware
- [VB1940 Eagle Camera - Leopard Imaging](https://leopardimaging.com/product/depth-sensing/stereoscopic-cameras/li-vb1940-stxxx-10gige/li-vb1940-vcl-st80-10gige-120h-poe/)

### Related Examples
- **Applications Overview**: `../` - Background on ROS2 and Holoscan integration
- **Simple Examples**: `../pubsub/` - Basic publisher/subscriber communication
- **Bridge Library**: `../../../operators/holoscan_ros2/` - Bridge implementation and headers

### Community and Support
- [Holoscan SDK GitHub](https://github.com/nvidia-holoscan/holoscan-sdk)
- [ROS2 Community](https://discourse.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
