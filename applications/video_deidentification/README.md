# Real-Time Face and Text Deidentification

<center> <img src="./docs/video_deid.gif" ></center>

This sample application demonstrates the use of face and text detection models to do real-time video deidentification.
Regions identified to be face or text are blurred out from the final image.

> **_NOTE:_** This application is a demonstration of real-time face and text deidentification and is not meant to be used in critical applications
that has zero error tolerance.  The models used in this sample application have limitations, e.g., in detecting faces and text that are
partially occluded, in low lighting situations, when there is motion blur, etc.

## Models

This application uses TAO PeopleNet model from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) for detecting faces.
The model is downloaded when building the application.

For text detection, this application uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) python library which uses Character Region Awareness for Text Detection [(CRAFT)](https://github.com/clovaai/CRAFT-pytorch).

## Data

This application downloads a pre-recorded video from [Pexels](https://www.pexels.com/video/young-traveler-walking-in-the-streets-of-milan-5271997/) when the application is built for use with this application.  Please review the [license terms](https://www.pexels.com/license/) from Pexels.

> **_NOTE:_** The user is responsible for checking if the dataset license is fit for the intended purpose.

## Input

This app currently supports three different input options:

1. v4l2 compatible input device (default, see V4L2 Support below)
2. pre-recorded video (see Video Replayer Support below)

## Run Instructions

## V4L2 Support

This application supports v4l2 compatible devices as input.  To run this application with your v4l2 compatible device,
please plug in your input device and run:
```sh
./holohub run video_deidentification
```

By default, this application expects the input device to be mounted at `/dev/video0`.  If this is not the case, please update
`applications/video_deidentification/video_deidentification.yaml` and set it to use the corresponding input device before
running the application.  You can also override the default input device on the command line by running:
```sh
./holohub run video_deidentification --run-args="--video_device /dev/video0"
```

## Video Replayer Support

If you don't have a v4l2 compatible device plugged in, you may also run this application on a pre-recorded video.
To launch the application using the Video Stream Replayer as the input source, run:

```sh
./holohub run video_deidentification --run-args="--source replayer"
```

### Known Issues

There is a known issue running this application on IGX w/ iGPU and on Jetson AGX (see [#500](https://github.com/nvidia-holoscan/holohub/issues/500)).
The workaround is to update the device to avoid picking up the libnvv4l2.so library.

```bash
cd /usr/lib/aarch64-linux-gnu/
ls -l libv4l2.so.0.0.999999
sudo rm libv4l2.so.0.0.999999
sudo ln -s libv4l2.so.0.0.0.0  libv4l2.so.0.0.999999
```
