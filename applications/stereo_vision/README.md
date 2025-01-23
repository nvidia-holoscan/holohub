# Stereo Vision

<p align="center">
  <img src="./images/plants.gif" alt="Holoscan Stereo Vision">
</p>

## Overview

A demo pipeline showcasing stereo disparity estimation and object detection.

## Description

This pipeline takes video from a stereo camera and estimates disparity using DNN ESS and object
detection using YOLO. The disparity maps and bounding boxes are displayed through Holoviz.

## Requirements

This application requires a V4L2 stereo camera or recorded stereo video as input. A video acquired from a StereoLabs ZED
camera is downloaded when running the `get_data_and_models.sh` script when building the application.
A script for obtaining the calibration for StereoLabs cameras is also provided.

### Camera Calibration

The default calibration will work for the sample video. If using a stereolabs camera the calibration
can be retrieved using `get_zed_calibration.py` and the devices serial number.

```sh
python3 get_zed_calibration.py -s [Serial Number]
```

### Input video

For the input video stream, either use a v4l2 stereo camera such as those produced by stereolabs or the recorded `stereo-plants.mp4` video. If using
recorded video this can be played using v4l2 loopback as described [here.](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/v4l2_camera#use-with-v4l2-loopback-devices)
The `stereo-plants.mp4` video is provided [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_stereo_video) and will be downloaded when running `get_data_and_models.sh` when following the instructions in the <b>Build and Run Instructions</b> section.

The source device in stereo_vision.yaml should be modified to match the device the v4l2 video is
using. This can be found using `v4l2-ctl --list-devices`.


## Models

This demo requires the ESS DNN Stereo Disparity availible from the NGC catalog for disparity and the
YOLOv8 onnx model for object detection. Both models are downloaded when you build the application.

### ESS DNN

The ESS engine files generated in this demo application is specific to TRT8.6; make sure
you build the devcontainer with a compatible `base_img` as shown in the <b>Build and Run Instructions</b> section.

### YOLOv8

For object detection, a YOLOv8 model from [Ultralytics](https://docs.ultralytics.com/models/yolov8/) is used
and exported to ONNX with non-max suppression plugin as mentioned [here](https://github.com/triple-Mu/YOLOv8-TensorRT).

## Build and Run Instructions

To build this application and download the necessary videos and models, run:
```sh
./dev_container build --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.4.0-dgpu --img holohub:stereo_vision
./dev_container launch --img holohub:stereo_vision
source applications/stereo_vision/scripts/get_data_and_models.sh data/stereo_vision
./run build stereo_vision
```

If you are using the recorded video as input, start the video playback outside the docker container.  The
following command will mount and stream the video to `/dev/video3` device.
```sh
sudo modprobe v4l2loopback video_nr=3 max_buffers=4
ffmpeg -stream_loop -1 -re -i data/stereo_vision/stereo-plants.mp4 -pix_fmt yuyv422 -f v4l2 /dev/video3
```

Return to the first terminal and run:
```sh
./run launch stereo_vision
```
