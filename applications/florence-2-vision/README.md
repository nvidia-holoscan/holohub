# 📷🤖 Florence-2

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
./dev_container build_and_run florence-2
```
Note: The first build will take **~1.5 hours** if you're on ARM64. This is largely due to building [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) since pre-built wheels are not distributed for ARM64 platforms.

## 💻 Supported Hardware
- IGX w/ dGPU
- x86 w/ dGPU
> Note: iGPU is not yet supported. This app can be built on iGPU devices and the florence2_app.py will run, however, the full QT UI app is not yet supported for iGPU.