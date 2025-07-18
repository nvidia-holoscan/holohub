# Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

This application demonstrates how to run the [Florence-2](https://arxiv.org/abs/2311.06242) models on a live video feed with the possibility of changing the task and optional prompt via a QT UI.



<p align="center">
  <img src="./demo.gif" alt="Holoscan VILA Live">
</p>

Note: This demo currently uses [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft), but any of the Florence-2 models should work as long as the correct URLs and names are used in [Dockerfile](./Dockerfile) and [config.yaml](./config.yaml):
- [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)
- [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft)
- [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)

## ⚙️ Setup Instructions
The app defaults to using the video device at `/dev/video0`

> Note: You can use a USB webcam as the video source, or an MP4 video by following the instructions for the [V4L2_Camera](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/v4l2_camera#use-with-v4l2-loopback-devices) example app.

To debug if this is the correct device download `v4l2-ctl`:
```bash
sudo apt-get install v4l-utils
```
To check for your devices run:
```bash
v4l2-ctl --list-devices
```
This command will output something similar to this:
```bash
NVIDIA Tegra Video Input Device (platform:tegra-camrtc-ca):
        /dev/media0

vi-output, lt6911uxc 2-0056 (platform:tegra-capture-vi:0):
        /dev/video0

Dummy video device (0x0000) (platform:v4l2loopback-000):
        /dev/video3
```
Determine your desired video device and edit the source device in [config.yaml](config.yaml)

## 🚀 Build and Run Instructions
From the Holohub main directory run the following command:
```bash
./holohub run florence-2-vision
```
Note: The first build will take **~1.5 hours** if you're on ARM64. This is largely due to building [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) since pre-built wheels are not distributed for ARM64 platforms.

## 💻 Supported Hardware
- IGX w/ dGPU
- x86 w/ dGPU
- IGX w/ iGPU and Jetson AGX supported with workaround<br>
  There is a known issue running this application on IGX w/ iGPU and on Jetson AGX (see [#500](https://github.com/nvidia-holoscan/holohub/issues/500)).
  The workaround is to update the device to avoid picking up the libnvv4l2.so library.

  ```bash
  cd /usr/lib/aarch64-linux-gnu/
  ls -l libv4l2.so.0.0.999999
  sudo rm libv4l2.so.0.0.999999
  sudo ln -s libv4l2.so.0.0.0.0  libv4l2.so.0.0.999999
  ```


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./holohub vscode florence-2-vision
```

This command will build and configure a Dev Container using a [Dockerfile](./Dockerfile) that is ready to run the application.

### VS Code Launch Profiles

There are two launch profiles configured for this application:

1. **(debugpy) florence-2-vision/python**: Launch florence-2-vision using a launch profile that enables debugging of Python code.
2. **(pythoncpp) florence-2-vision/python**: Launch florence-2-vision using a launch profile that enables debugging of Python and C++ code.
