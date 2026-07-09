# Holoscan ROS2 VB1940 (Eagle) Camera on Jetson Thor

## Overview

This application is a port of the [VB1940 camera example](../vb1940/) to NVIDIA Jetson AGX Thor, containing both publisher and subscriber components in a single structure. The application shows how to:

- Capture images from a VB1940 (Eagle) camera (Holoscan Sensor Bridge) using Holoscan (Publisher mode)
- Process camera data through a complete pipeline
- Publish processed images to ROS2 topics for visualization or further processing
- Subscribe to and visualize camera streams (Subscriber mode)

The application uses **modes** to run either the publisher or subscriber component, with both C++ implementations organized under a unified `cpp/` directory.

Unlike the original `vb1940` example (which uses `RoceReceiverOp` and requires a ConnectX NIC with RDMA/RoCE support), this application receives camera data with the **Linux socket receiver** (`LinuxReceiverOp`) over the standard Linux network stack. On NVIDIA Jetson AGX Thor this uses the built-in **MGBE 10GbE ports** — no ConnectX NIC or RDMA/RoCE support is required.

## File Structure

```text
applications/holoscan_ros2/thor_vb1940/
├── CMakeLists.txt          # Top-level build configuration
├── Dockerfile              # Container build (ROS2 Jazzy + holoscan-sensor-bridge)
├── README.md               # This file
└── cpp/                    # C++ implementation
    ├── CMakeLists.txt      # C++ build configuration
    ├── metadata.json       # Application metadata with modes
    ├── convert_16bit_to_8bit_kernel.h    # CUDA conversion kernel header
    ├── convert_16bit_to_8bit_kernel.cu   # CUDA conversion kernel
    ├── thor_vb1940_publisher.cpp    # Publisher implementation
    └── thor_vb1940_subscriber.cpp   # Subscriber implementation
```

## Prerequisites

- **NVIDIA Holoscan SDK** v3.0 or later (container base image: `holoscan:v4.4.0-cuda13`)
- **ROS2 Jazzy** (all examples and Dockerfiles are tested with Jazzy; other distributions may work but are not tested)
- **Docker** (with NVIDIA Container Toolkit)
- **NVIDIA GPU drivers** (suitable for your hardware and Holoscan SDK)

For VB1940 camera capture (publisher mode), you'll also need:

- Holoscan Sensor Bridge (Hololink) board with a VB1940 (Eagle) camera module
- Proper network configuration for camera communication (see below)

Targeted at NVIDIA Jetson AGX Thor (JetPack 7.x, `6.8.12-*-tegra` kernel, aarch64).

## Network Setup

Before running the VB1940 publisher, ensure proper network configuration:

- Default Hololink board IP: `192.168.0.2`
- Connect the sensor bridge to one of the Thor MGBE ethernet ports and give that interface an address on the same subnet (e.g. `192.168.0.101/24`)
- Verify connectivity with `ping 192.168.0.2`
- The Linux socket receiver benefits from a large kernel receive buffer, e.g.:

  ```bash
  sudo sysctl -w net.core.rmem_max=31326208
  ```

No InfiniBand/RoCE (IBV) device configuration is needed.

## Building the Application

Build the unified VB1940 camera application:

```bash
./holohub build thor_vb1940
```

The build uses the public [holoscan-sensor-bridge](https://github.com/nvidia-holoscan/holoscan-sensor-bridge) repository, built with `-DHOLOLINK_BUILD_ROCE=OFF`.

## Application Modes

The thor_vb1940 application uses **modes** to handle the publisher and subscriber components. You can list available modes using:

```bash
./holohub modes thor_vb1940 --language cpp
```

Available modes:

- **publisher**: Captures and publishes VB1940 camera images (default mode)
- **subscriber**: Receives and visualizes camera images

## Running the Application

### Publisher Mode (`thor_vb1940_publisher.cpp`)

Captures images from a VB1940 camera using Holoscan and publishes to ROS2:

- Receives CSI frames over UDP (Linux socket receiver / Thor MGBE)
- Processes images through a complete pipeline:
  - CSI to Bayer conversion
  - Image processing
  - Bayer demosaicing
  - GPU 16-bit to 8-bit RGB conversion
- Publishes processed images to ROS2 topic `vb1940/image`
- Supports various camera modes and configurations

**Usage:**

```bash
# Run publisher (default mode)
./holohub run thor_vb1940 [options]
# Or explicitly specify publisher mode:
./holohub run thor_vb1940 publisher [options]
```

**Publisher Options:**

- `--camera-mode`: VB1940 mode (default: 2560x1984 30FPS)
- `--frame-limit`: Exit after publishing specified number of frames
- `--hololink`: IP address of Hololink board (default: 192.168.0.2)

### Subscriber Mode (`thor_vb1940_subscriber.cpp`)

Subscribes to camera images from ROS2 and visualizes them:

- Subscribes to the `vb1940/image` ROS2 topic
- Receives and processes the images
- Visualizes images using Holoviz
- Supports headless and fullscreen modes

**Usage:**

```bash
./holohub run thor_vb1940 subscriber [options]
```

**Subscriber Options:**

- `--headless`: Run in headless mode
- `--fullscreen`: Run in fullscreen mode

## Additional Resources

### Documentation

- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
- [Holoscan Sensor Bridge Documentation](https://docs.nvidia.com/holoscan/sensor-bridge/latest/)
- [ROS2 Jazzy Documentation](https://docs.ros.org/en/jazzy/index.html)

### Related Examples

- **Applications Overview**: `../` - Background on ROS2 and Holoscan integration
- **Simple Examples**: `../pubsub/` - Basic publisher/subscriber communication
- **VB1940 Examples**: `../vb1940/` - Original RoCE/ConnectX based variant
- **IMX274 Examples**: `../imx274/` - Linux receiver variant for the IMX274 camera
- **Bridge Library**: `../../../operators/holoscan_ros2/` - Bridge implementation and headers

### Community and Support

- [Holoscan SDK GitHub](https://github.com/nvidia-holoscan/holoscan-sdk)
- [Holoscan Sensor Bridge GitHub](https://github.com/nvidia-holoscan/holoscan-sensor-bridge)
- [ROS2 Community](https://discourse.ros.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
