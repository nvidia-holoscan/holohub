# H.264 Video Decode Reference Application

This is a minimal reference application demonstrating usage of H.264 video
decode operator. This application makes use of H.264 elementary stream reader
operator for reading H.264 elementary stream input and uses Holoviz operator
for rendering decoded data to the native window.

_The H.264 video decode operator does not adjust framerate as it reads the elementary
stream input. As a result the video stream will be displayed as quickly as the decoding can be
performed. This feature will be coming soon to a new version of the operator._

## Requirements

This application is configured to use H.264 elementary stream from endoscopy
sample data as input. To use any other stream, the filename / path for the
input file can be specified in the 'h264_video_decode.yaml' file.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

Unzip the sample data:

```
unzip holoscan_endoscopy_sample_data_20230128.zip -d <data_dir>
```

## Building the application

Please refer to the top level Holohub README.md file for information on how to build this application.

## Running the application

Run the application `h264_video_decode` in the binary directory.

```bash
cd <build_dir>/applications/h264_video_decode && ./h264_video_decode --data <HOLOHUB_DATA_DIR>/endoscopy
```
