# Polyp Detection

## Build Docker

```Bash
docker build -t nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu-polyp-det -f Dockerfile .
```

## Run Application

### Run Docker

Please mount the directory that contains `holohub/` into `/colon`. For example, my folder is under `/raid/colon_reproduce`.

```Bash
docker run --rm -it --gpus=all --ipc=host -p 8888:8888 -v /raid/colon_reproduce:/colon nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu-polyp-det
```

### Generate TensorRT model

```Bash
trtexec --onnx=rtdetrv2_timm_r50_nvimagenet_pretrained.onnx --saveEngine=rt_detrv2_timm_r50_nvimagenet_pretrained_demo.trt \
--minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
--optShapes=images:128x3x640x640,orig_target_sizes:128x2 \
--maxShapes=images:128x3x640x640,orig_target_sizes:128x2 \
--allowGPUFallback
```

### Grab example video

(todo) need to upload after license permisison

### Convert Video into GXF Entities

```Bash
apt-get update && apt-get install -y ffmpeg
git clone https://github.com/nvidia-holoscan/holoscan-sdk.git
ffmpeg -i <prepared video>.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python holoscan-sdk/scripts/convert_video_to_gxf_entities.py --width 1164 --height 1034 --channels 3 --framerate 30 --directory /colon/holohub/data/polyp_detection/
```

### Run application

```Bash
cd /colon/holohub/applications/polyp_detection
python polyp_detection.py --data /colon/holohub/data/polyp_detection/
```
