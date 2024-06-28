# Endoscopy Out of Body Detection Application

This application performs endoscopy out of body detection. The application classifies if the input frame is inside the body or out of the body. If the input frame is inside the body, application prints `Likely in-body`, otherwise `Likely out-of-body`. Each likelihood is accompanied with a confidence score. If the analytics is enabled, the output for each input frame is instead exported to the given csv file.

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
data
└── endoscopy_out_of_body_detection
    ├── LICENSE.md
    ├── out_of_body_detection.onnx
    ├── sample_clip_out_of_body_detection.gxf_entities
    ├── sample_clip_out_of_body_detection.gxf_index
    └── sample_clip_out_of_body_detection.mp4
```

## Building the application

Please refer to the top level Holohub README.md file for information on how to build this application.

Additionally, if the Holoscan SDK version is 0.5 or lower, following code changes must be made in the application:

* In main.cpp: `#include <holoscan/operators/inference/inference.hpp>` is replaced with `#include <holoscan/operators/multiai_inference/multiai_inference.hpp>`
* In main.cpp: `#include <holoscan/operators/inference_processor/inference_processor.hpp>` is replaced with `#include <holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp>`
* In main.cpp: `ops::InferenceOp` is replaced with `ops::MultiAIInferenceOp`
* In main.cpp: `ops::InferenceProcessorOp` is replaced with `ops::MultiAIPostprocessorOp`
* In CMakeLists.txt: update the holoscan SDK version from `0.6` to `0.5`
* In CMakeLists.txt: `holoscan::ops::inference` is replaced with `holoscan::ops::multiai_inference`
* In CMakeLists.txt: `holoscan::ops::inference_processor` is replaced with `holoscan::ops::multiai_postprocessor`

## Running the application

In your `build` directory, run

```bash
applications/endoscopy_out_of_body_detection/endoscopy_out_of_body_detection --data ../data/endoscopy_out_of_body_detection
```

## Enable analytics

The endoscopy out of body detetction application supports exporting output to the comma separated value (CSV) files. This data can later be used by analytics applications. The analytics data generation can be enabled by setting `enable_analytics` flag in the config file `<build_dir>/applications/endoscopy_out_of_body_detection/endoscopy_out_of_body_detection.yaml` to `true`.

The data root directory can be specified using the environment variable `HOLOSCAN_ANALYTICS_DATA_DIRECTORY`. If not specified, it defaults to the current directory. The CSV data file name can be specified using the environment variable `HOLOSCAN_ANALYTICS_DATA_FILE_NAME`. If not specified, it defaults to the name `data.csv`. All the generated data will be stored inside a directory with the same name as the application name that is passed in the configuration file. On each run, a new directory inside the data directory will be created and a new data file will be created inside it. Each new data directory will be named with the current timestamp.
