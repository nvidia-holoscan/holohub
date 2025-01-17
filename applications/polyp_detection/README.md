# Polyp Detection

This application demonstrates how to run polyp detection models on live video in real-time.

The model: [RT-DETR](https://github.com/lyuwenyu/RT-DETR) is trained on the [REAL-Colon](https://www.nature.com/articles/s41597-024-03359-0) dataset.

## Setup Instructions

### Build Docker and Run

```Bash
docker build -t holoscan:polyp-det -f Dockerfile .
```

Please mount the directory that contains `holohub/` into `/colon_workspace`. For example, my folder is under `/raid/colon_reproduce`.

```Bash
docker run --rm -it --gpus=all --ipc=host -p 8888:8888 -v /raid/colon_reproduce:/colon_workspace holoscan:polyp-det
```

### Get ONNX model

(TODO) need to upload after license permission

### Generate TensorRT model

```Bash
trtexec --onnx=<path-to-onnx>/polyp_det_model.onnx --saveEngine=polyp_det_model.trt \
--minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
--optShapes=images:32x3x640x640,orig_target_sizes:32x2 \
--maxShapes=images:32x3x640x640,orig_target_sizes:32x2 \
--allowGPUFallback
```

### Get Example Video

(TODO) need to upload after license permission

### Convert Video into GXF Entities

```Bash
apt-get update && apt-get install -y ffmpeg
git clone https://github.com/nvidia-holoscan/holoscan-sdk.git
ffmpeg -i <prepared video>.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python holoscan-sdk/scripts/convert_video_to_gxf_entities.py --width 1164 --height 1034 --channels 3 --framerate 30 --directory /colon/holohub/data/polyp_detection/
```

### Run application

```Bash
cd /colon_workspace/holohub/applications/polyp_detection
python polyp_detection.py --data /colon_workspace/holohub/data/polyp_detection/
```
