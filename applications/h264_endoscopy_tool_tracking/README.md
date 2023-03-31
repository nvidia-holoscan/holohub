# H264 Endoscopy Tool Tracking Application

The application showcases how to use H.264 video source as input to and output
from the Holoscan pipeline. This application is a modified version of Endoscopy
Tool Tracking reference application in Holoscan SDK that supports H.264
elementary streams as the input and output.

_The H.264 video decode operator does not adjust framerate as it reads the elementary
stream input. As a result the video stream will be displayed as quickly as the decoding can be
performed. This feature will be coming soon to a new version of the operator._

## Requirements

This application is configured to use H.264 elementary stream from endoscopy
sample data as input. The output of the pipeline is again recorded to a H.264
elementary stream on the disk, file name / path for this can be specified in
the 'h264_endoscopy_tool_tracking.yaml' file.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

Unzip the sample data:

```
unzip holoscan_endoscopy_sample_data_20230128.zip -d <data_dir>
```

## Building the application

Please refer to the top level Holohub README.md file for information on how to build this application.

## Running the application

Run the application `h264_endoscopy_tool_tracking` in the binary directory.

```bash
cd <build_dir>/applications/h264_endoscopy_tool_tracking/ \
  && ./h264_endoscopy_tool_tracking --data <HOLOHUB_DATA_DIR>/endoscopy
```
