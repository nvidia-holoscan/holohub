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

For the input video stream, either use a v4l2 stereo camera such as those produced by stereolabs or included recorded video.
The `stereo-plants.mp4` video is provided [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_stereo_video) and will be downloaded and converted to the necessary format when building the application.

The source device in `stereo_vision.yaml` should be modified to match the device the v4l2 video is
using. This can be found using `v4l2-ctl --list-devices`.


## Models

This demo requires the ESS DNN Stereo Disparity available from the NGC catalog for disparity and the
YOLOv8 onnx model for object detection. Both models are downloaded when you build the application.

### ESS DNN

The ESS engine files generated in this demo application is specific to TRT8.6; make sure
you build the devcontainer with a compatible `base_img` as shown in the <b>Build and Run Instructions</b> section.

### YOLOv8

For object detection, a YOLOv8 model from [Ultralytics](https://docs.ultralytics.com/models/yolov8/) is used
and exported to ONNX with non-max suppression plugin as mentioned [here](https://github.com/triple-Mu/YOLOv8-TensorRT).

## Build and Run Instructions

Run the following command to build and run application using the recorded video:
```sh
./dev_container build_and_run stereo_vision --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.4.0-dgpu
```

To run the application using a v4l2 compatible stereo camera, run:
```sh
./dev_container build_and_run stereo_vision --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.4.0-dgpu --run_args "--source v4l2"
```
