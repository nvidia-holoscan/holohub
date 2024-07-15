# TAO PeopleNet Detection Model on V4L2 Video Stream

Use the TAO PeopleNet available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) to detect faces and people in a V4L2 supported video stream. HoloViz is used to draw bounding boxes around the detections.

## Model

This application uses the TAO PeopleNet model from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) for face and person classification.
The model is downloaded when building the application.

## Data

This application downloads a pre-recorded video from [Pexels](https://www.pexels.com/video/a-woman-showing-her-ballet-skill-in-turning-one-footed-5385885/) when the application is built for use with this application.  Please review the [license terms](https://www.pexels.com/license/) from Pexels.

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
./dev_container build_and_run tao_peoplenet
```

By default, this application expects the input device to be mounted at `/dev/video0`.  If this is not the case, please update
`applications/tao_peoplenet/tao_peoplenet.yaml` and set it to use the corresponding input device before
running the application.  You can also override the default input device on the command line by running:
```sh
./dev_container build_and_run tao_peoplenet --run_args "--video_device /dev/video0"
```

## Video Replayer Support

If you don't have a v4l2 compatible device plugged in, you may also run this application on a pre-recorded video.
To launch the application using the Video Stream Replayer as the input source, run:

```sh
./dev_container build_and_run tao_peopelnet --run_args "--source replayer"
```
