# Polyp Detection

This application demonstrates how to run polyp detection models on live video in real-time.

The model: [RT-DETR](https://github.com/lyuwenyu/RT-DETR) is trained on the [REAL-Colon](https://www.nature.com/articles/s41597-024-03359-0) dataset.

Compared to the `SSD` object detection model described in the [paper](https://www.nature.com/articles/s41597-024-03359-0), `RT-DETR` demonstrates improvements. The table below shows metrics for SSD obtained from Table 3 of the paper, and metrics for RT-DETR calculated on the same test set (using all test images from the `REAL-Colon` dataset).

<div style="text-align: center;">

| Method  | MAP@0.5 | MAP@0.5:0.95 |
|---------|---------|--------------|
| SSD     | 0.338   | 0.216        |
| RT-DETR | 0.452   | 0.301        |

</div>


## Run Instructions

### Step 1: Build and Launch Container

From the Holohub main directory run the following command to build the container:

```Bash
./dev_container build
```

Launch the container:

```Bash
./dev_container launch
```

### Step 2: Download the trained ONNX Model 

(TODO) need to upload after license permission.

Download the ONNX Model `rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune_bhwc.onnx` into the data path, for example: `./data`.

### Step 3: Prepare the Video

(TODO) need to upload sample video.
    
Video files need to be converted into a GXF replayable tensor format to be used as stream inputs. This step has already been done for the sample video. To do so for your own video data, we provide a [utility script on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) named ```convert_video_to_gxf_entities.py```. This script should yield two files in .gxf_index and .gxf_entities formats, which can be used as inputs with Holoscan.

Assume the video is in `./path-to-video/sample.mp4`, width and height are 1164 and 1034, follow below command to convert the video:
Follow below procedures to convert the video.

```Bash
git clone https://github.com/nvidia-holoscan/holoscan-sdk.git
ffmpeg -i /path-to-video/sample.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 |  python holoscan-sdk/scripts/convert_video_to_gxf_entities.py --width 1164 --height 1034 --channels 3 --framerate 30 --directory /path-to-video/
```

### Step 4: Run application

Ensure that the `rtdetrv2_timm_r50_nvimagenet_pretrained_neg_finetune_bhwc.onnx` file is located in the directory specified by the `data` argument.
Verify that the generated video files (`.gxf_index` and `.gxf_entities` files) are in the directory specified by the `video_dir` argument.
Specify the correct video width and height using the `video_size` argument.

```Bash
python polyp_detection.py --data /path-to-model/ --video_dir /path-to-video/ --video_size "(1164, 1034)"
```
