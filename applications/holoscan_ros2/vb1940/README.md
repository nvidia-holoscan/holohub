# Holoscan ROS2 VB1940 (Eagle) Camera Examples

## Overview
The VB1940 (Eagle) Camera examples demonstrate advanced usage with real camera hardware. These examples show how to:
- Capture images from a VB1940 (Eagle) camera using Holoscan
- Process camera data through a complete pipeline
- Publish processed images to ROS2 topics for visualization or further processing
- Subscribe to and visualize camera streams

**Important:** These examples require access to NVIDIA's internal hololink repository and VB1940 camera hardware.

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

## Network Setup
Before running VB1940 examples, ensure proper network configuration:
- Default Hololink board IP: `192.168.0.2`
- Ensure your host can reach the Hololink board
- Verify IBV (InfiniBand Verbs) device configuration if using custom hardware

## Applications

### 1. Publisher (`vb1940_publisher.cpp`)
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

### 2. Subscriber (`vb1940_subscriber.cpp`)
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

## Troubleshooting

### SSH Access for Hololink Repository
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

### Common Error Messages and Solutions

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
./isaac-hub build holoscan_ros2_vb1940_publisher --build-args="--ssh default" --docker-opts "--runtime=runc --gpus all"
```

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
