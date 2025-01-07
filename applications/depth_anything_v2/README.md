# Depth Anything V2
<div align="center">
    <img src="./docs/depth.gif" width="500" height="363">
</div>

This application uses the Depth Anything V2 model for monocular depth estimation.  <b>Monocular Depth Estimation</b> refers to the task of predicting the distance of objects in a scene from a single 2D image captured by a standard camera.

## Model

This application uses the Depth Anything V2 model from [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation.
The model is downloaded when building the Docker image.

> **_NOTE:_** The user is responsible for checking if the model license is suitable for the intended purpose.

## Data

This application downloads a pre-recorded video from [Pexels](https://www.pexels.com/video/a-woman-running-on-a-pathway-5823544/) when the application is built.  Please review the [license terms](https://www.pexels.com/license/) from Pexels.

> **_NOTE:_** The user is responsible for ensuring the dataset license is suitable for the intended purpose.

## Input

This app currently supports two input options:

1. v4l2 compatible input device (default; see <b>V4L2 Support</b> below)
2. Pre-recorded video (see <b>Video Replayer Support</b> below)

## Run Instructions

### V4L2 Support

This application supports v4l2 compatible devices as input.  To run this application with your v4l2 compatible device,
please plug in your input device and run:
```sh
./dev_container build_and_run depth_anything_v2
```

By default, this application expects the input device to be mounted at `/dev/video0`.  If this is not the case, update
`applications/depth_anything_v2/depth_anything_v2.yaml` file to set the corresponding input device before
running the application.  You can also override the default input device on the command line by running:
```sh
./dev_container build_and_run depth_anything_v2 --run_args "--video_device /dev/video0"
```

### Video Replayer Support

If you don't have a v4l2 compatible device plugged in, you can also run this application on a pre-recorded video.
To launch the application using the Video Stream Replayer as the input source, run:

```sh
./dev_container build_and_run depth_anything_v2 --run_args "--source replayer"
```

### Display Modes

This application has multiple display modes which you can toggle through using the left mouse button.

* original: output the original image from input source
* depth: output the color depthmap based on the depthmap returned from Depth Anything V2 model
* side-by-side: output a side-by-side view of the original image next to the color depthmap
* interactive: allow user 

In interactive mode, the middle or right mouse button can be used to modify the ratio of original image vs color depthmap is shown.


## Acknowledgement

This project is based on the following projects:
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) - Depth Anything V2
- [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt) - Depth Anything TensorRT CLI

## Known Issues

There is a known issue running this application on IGX w/ iGPU and on Jetson AGX (see [#500](https://github.com/nvidia-holoscan/holohub/issues/500)).
The workaround is to update the device to avoid picking up the libnvv4l2.so library.

```bash
cd /usr/lib/aarch64-linux-gnu/
ls -l libv4l2.so.0.0.999999
sudo rm libv4l2.so.0.0.999999
sudo ln -s libv4l2.so.0.0.0.0  libv4l2.so.0.0.999999
```
