# Body Pose Estimation App
<div align="center">
    <img src="./docs/1.png" width="300" height="375">
    <img src="./docs/2.png" width="300" height="375">
    <img src="./docs/3.png" width="300" height="375">
</div>

Body pose estimation is a computer vision task that involves recognizing specific points on the human body in images or videos.
A model is used to infer the locations of keypoints from the source video which is then rendered by the visualizer. 

## Model

This application uses YOLOv8 pose model from [Ultralytics](https://docs.ultralytics.com/tasks/pose/) for body pose estimation.
The model is downloaded when building the application.

## Requirements

This application uses a v4l2 compatible device as input.  Please plug in your input device (e.g., webcam) and ensure that `applications/body_pose_estimation/body_pose_estimation.yaml` is set to the corresponding device.  The default input device is set to `/dev/video0`.

## Run Instructions

Run the following commands to start the body pose estimation application:
```sh
./dev_container build --docker_file applications/body_pose_estimation/Dockerfile --img holohub:bpe
./dev_container launch --img holohub:bpe                                                         
./run build body_pose_estimation
./run launch body_pose_estimation
```
