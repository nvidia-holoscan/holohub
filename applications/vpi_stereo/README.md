# VPI Stereo Vision

<p align="center">
  <img src="./images/vpi_stereo.gif" alt="Stereo vision using VPI">
</p>

## Overview

Demo pipeline showing stereo disparity estimation using the
Vision Programming Interface [VPI](https://developer.nvidia.com/embedded/vpi).

## Description

This pipeline takes video from a stereo camera and uses VPI's
[stereo disparity estimation algorithm](https://docs.nvidia.com/vpi/algo_stereo_disparity.html).
The input video and estimate disparity map are displayed using Holoviz.

The application will select accelerator backends if available (OFA, PVA and VIC). This demonstrates
how VPI can be used to offload stereo disparity processing from the GPU on supported devices such as
NVIDIA IGX, AGX, or NX platforms.

## Input Video

Requires a V4L2 stereo camera or recorded stereo video. See the [stereo_vision](../stereo_vision/)
demo application for more information on how to obtain a stereo sample video, stereo camera
calibration data, and how to set up a v4l2 loopback device. The default configuration file for this
application uses /dev/video3 as input device.

## Requirements

This demo requires VPI version 3.2 or greater. The included Dockerfile will install VPI and its
dependencies for either an amd64 target (with discrete NVIDIA GPU), or arm64 target (NVIDIA IGX,
AGX, or NX).

## Build and Run Instructions

```sh
./dev_container build_and_run vpi_stereo
```
