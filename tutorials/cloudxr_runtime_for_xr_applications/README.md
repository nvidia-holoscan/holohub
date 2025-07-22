# Setting up CloudXR Runtime with Holoscan XR Applications

This tutorial demonstrates how to leverage **NVIDIA CloudXR Runtime** to stream Holoscan XR applications to XR devices like Apple Vision Pro. 

CloudXR enables you to run computationally intensive applications (medical volume rendering, AI inference) on powerful server hardware while delivering high-quality, low-latency XR experiences to XR headsets.

**What you'll accomplish:**
- Set up CloudXR Runtime as a bridge between Holoscan applications and XR devices
- Configure Apple Vision Pro as an XR client
- Successfully stream and experience immersive 3D content from XR applications to Apple Vision Pro

<p align="center">
  <img src="../../applications/xr_gsplat/doc/gsplat-demo.gif" alt="Gsplat Demo on AVP">
</p>

## Table of Contents
- [System Requirements](#system-requirements)
- [Run CloudXR Runtime Container with Docker](#run-cloudxr-runtime-container-with-docker)
- [Run Your XR Application](#run-your-xr-application)
- [Set up Apple Vision Pro](#set-up-apple-vision-pro)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Apple Vision Pro
- visionOS 2.0 or later
- Apple M3 Pro chip with an 11-core CPU (minimum 5 performance cores and 6 efficiency cores)
- 16GB unified memory
- 256GB SSD storage

### Apple Silicon Mac
Required for building the Isaac XR Teleop Sample Client App for Apple Vision Pro with Xcode:
- macOS Sonoma 14.5 or later
- Xcode development environment

## Run CloudXR Runtime Container with Docker

The CloudXR Runtime runs in a Docker container on the same machine as your application, serving as a bidirectional bridge that streams sensor data from the XR device to your application and display content back to the XR device.

### Step 1: Configure Firewall
Ensure your firewall allows connections to the ports used by CloudXR:

```bash
sudo ufw allow 47998:48000,48005,48008,48012/udp
sudo ufw allow 48010/tcp
```

### Step 2: Create Shared Directory
Create a shared directory for communication between XR applications and the CloudXR Runtime:

```bash
mkdir -p $(pwd)/openxr
```

This directory will be used for temporary cache files and runtime communication.

### Step 3: Start CloudXR Runtime Container
Launch the CloudXR Runtime container, mounting the shared directory:

```bash
docker run -it --rm --name cloudxr-runtime \
    --user $(id -u):$(id -g) \
    --gpus=all \
    -e "ACCEPT_EULA=Y" \
    --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
    -p 48010:48010 \
    -p 47998:47998/udp \
    -p 47999:47999/udp \
    -p 48000:48000/udp \
    -p 48005:48005/udp \
    -p 48008:48008/udp \
    -p 48012:48012/udp \
    nvcr.io/nvidia/cloudxr-runtime:0.1.0-isaac
```

The container should start and wait for client connections.

## Run Your XR Application
This tutorial can work across XR applications in holohub, including [volume_rendering_xr](../../applications/volume_rendering_xr/), [xr_gsplat](../../applications/xr_gsplat/), and [xr_holoviz](../../applications/xr_holoviz/).

### Step 1: Build XR Application
By default, you can build your chosen Holoscan XR application using the command:
```shell
# from the root directory of holohub
./holohub run-container <app-name>
```
Please refer to the specific application's README (for example, [xr_gsplat](../../applications/xr_gsplat/README.md)) for detailed build and run instructions.

### Step 2: Configure Environment Variables
Inside your application container or environment, set up the CloudXR runtime variables:

```bash
export XDG_RUNTIME_DIR=$(pwd)/openxr/run
export XR_RUNTIME_JSON=$(pwd)/openxr/share/openxr/1/openxr_cloudxr.json
```

These environment variables configure your application to use the CloudXR runtime instead of local OpenXR runtimes.

### Step 3: Launch Application
Inside your application container, launch your XR application using:
```shell
./holohub run <app-name>
```

## Set up Apple Vision Pro

### Step 1: Install Client Application
1. Follow the instructions at [isaac-xr-teleop-sample-client-apple](https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple) to install the client app for Apple Vision Pro
2. Build and deploy the app to your device using Xcode

> **Note**: The Isaac XR Teleop Sample Client app is primarily designed for teleoperation with Isaac applications. In this tutorial, we're using it solely as a CloudXR client to connect to and stream content from the host machine.
### Step 2: Connect to Host
1. Open the **Isaac XR Teleop Sample Client** on your Apple Vision Pro
2. You should see a connection UI interface
3. Enter the **IP address** of the host machine running your XR application
4. Tap **Connect**

### Step 3: Experience XR Content
Once connected successfully, you should see your 3D application content streaming live in the headset with full immersive experience.

## Troubleshooting

### Connection Issues
If you experience connection problems between CloudXR runtime and Apple Vision Pro, verify the following:

**Basic Checks:**
1. **CloudXR container is running** on the host machine
2. After clicking `Connect` on Apple Vision Pro, the CloudXR container should log that **a client has connected**

**Network Configuration:**
- **Firewall settings**: Ensure the host firewall is properly configured (see Step 1 above)
- **Network connectivity**: Verify that Apple Vision Pro and the host machine are on the same network
- **IP address**: Confirm the Apple Vision Pro teleop app is pointing to the correct host IP address

### Common Error Messages

#### "Out of buffer space when trying to write format type spacer"
- Disconnect and reconnect WiFi on host machine


## References

This tutorial is largely based on the [Isaac Lab CloudXR Teleoperation Guide](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#run-isaac-lab-with-the-cloudxr-runtime).