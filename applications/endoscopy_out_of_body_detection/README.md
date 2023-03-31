# Endoscopy Out of Body Detection Application

This application performs endoscopy out of body detection. The application classifies if the input frame is inside the body or out of the body. If the input frame is inside the body, application prints `Likely in-body`, otherwise `Likely out-of-body`. Each likelihood is accompanied with a confidence score. 

__Note: there is no visualization component in the application.__

`endoscopy_out_of_body_detection.yaml` is the configuration file. Input video file is converted into GXF tensors and the name and location of the GXF tensors are updated in the `basename` and the `directory` field in `replayer`.

## Data

__Note: the data is automatically downloaded and converted when building. If you need to manually convert the data follow the following steps.__


* Endoscopy out of body detection model and the sample dataset is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/endoscopy_out_of_body_detection)
  After downloading the data in mp4 format, it must be converted into GXF tensors.
* Script for GXF tensor conversion (`convert_video_to_gxf_entities.py`) is available with the Holoscan SDK, and can be accessed [here](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts)

### Unzip and convert the sample data:

```
# unzip the downloaded sample data
unzip [FILE].zip -d <data_dir>

# convert video file into tensor
ffmpeg -i <INPUT_VIDEO_FILE> -fs 900M -pix_fmt rgb24 -f rawvideo pipe:1 | python convert_video_to_gxf_entities.py --width 256 --height 256 --channels 3 --framerate 30

# where <INPUT_VIDEO_FILE> is one of the downloaded MP4 files: OP1-out-2.mp4, OP4-out-8.mp4 or OP8-out-4.mp4.
```

Move the model file and converted video tensor into a directory structure similar to the following:

```bash
data/
├── out_of_body_detection.onnx
├── sample_clip_out_of_body_detection.mp4
└── tensor.gxf_entities
└── tensor.gxf_index
```

## Building the application

Please refer to the top level Holohub README.md file for information on how to build this application.

## Running the application

```bash
# ensure the current working directory contains the <data_dir>.
<build_dir>/endoscopy_out_of_body
```
