# Yolo Object Detection

This project is aiming to provide basic guidance to deploy Yolo-based model to Holoscan SDK as "Bring Your Own Model"

<div align="center">
    <img src="./docs/meeting.gif" width="500" height="363">
</div>

## Supported Platforms

The demo [container environment](Dockerfile) supports x86 or arm64 platforms with a discrete GPU.

Users targeting a Jetson platform (integrated GPU) may need to install [Jetson PyTorch packages](https://pypi.jetson-ai-lab.io/)
in their demo environment.

## Model
* Yolo v8 model: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
* Yolo v8 export repository: https://github.com/triple-Mu/YOLOv8-TensorRT

In this application example, we use the YOLOv8s model, which is converted to ONNX format using the repository mentioned above. Note that if you provide your own ONNX model, ensure it includes the EfficientNMS_TRT layer. You can verify it using [Netron](https://netron.app/). Additionally, we employ the `graph_surgeon.py` script to modify the input shape. For more details on this script, refer to [graph_surgeonpy](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#graph_surgeonpy).

The detailed process is documented in the `CMakeLists.txt` file.


## Input Source

This app currently supports two input options:

1. v4l2 compatible input device
2. Pre-recorded video

## Data

This application downloads a pre-recorded video from [Pexels](https://www.pexels.com/video/a-group-of-people-introduces-oneself-to-each-other-by-shaking-hands-3252919/) when the application is built.  Please review the [license terms](https://www.pexels.com/license/) from Pexels.

## Run

Build and launch container. Note that this will use a v4l2 input source as default.

```
./holohub run yolo_model_deployment
```

### Video Replayer Support

If you don't have a v4l2 compatible device plugged in, you can also run this application on a pre-recorded video. To launch the application using the Video Stream Replayer as the input source, run:

```
./holohub run yolo_model_deployment --run-args="--source replayer"
```

### Configuration

For application configuration, please refer to the `yolo_detection.yaml`.
