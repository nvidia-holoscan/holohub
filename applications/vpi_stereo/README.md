# VPI Stereo Vision

## Overview

Demo pipeline showing stereo disparity estimation using the
Vision Programming Interface [VPI](https://developer.nvidia.com/embedded/vpi).

## Description

This pipeline takes video from a stereo camera and uses VPI's
[stereo disparity estimation algorithm](https://docs.nvidia.com/vpi/algo_stereo_disparity.html).
The input video and estimate disparity map are displayed using Holoviz.

## Input Video

Requires a V4L2 stereo camera or recorded stereo video. See the [stereo_vision](../stereo_vision/)
demo application for more information on how to obtain a stereo sample video, stereo camera
calibration data, and how to set up a v4l2 loopback device. The default configuration file for this
application uses /dev/video3 as input device.

## Requirements

This demo requires VPI version 3.2 or greater. The included Dockerfile will install VPI and its
dependencies for either an x86_64 host (with NVIDIA GPU), or aarch64 host (NVIDIA IGX, NX or AGX).

## Build and Run Instructions

```sh
./dev_container build_and_run vpi_stereo --docker_file ./applications/vpi_stereo/Dockerfile
```

### Troubleshooting

If you see an error like this on IGX/AGX/NX platform:
```
Caught signal 11 (Segmentation fault: address not mapped to object at address 0x1c09)
```
It could be due to a known issue with Holoscan container version less than 2.8. To workaround, add
this argument to the build_and_run command: `--container_args " -e LIBV4L2_ENABLE_RTLD_NODELETE=1"`
