# H.264 Video Decode Reference Application

This is a minimal reference application demonstrating usage of H.264 video
decode operators. This application makes use of H.264 elementary stream reader
operator for reading H.264 elementary stream input and uses Holoviz operator
for rendering decoded data to the native window.

_The H.264 video decode operators do not adjust framerate as it reads the
elementary stream input. As a result the video stream can be displayed as
quickly as the decoding can be performed. This application uses
`PeriodicCondition` to play video at the same speed as the source video._

## Requirements

This application is configured to use H.264 elementary stream from endoscopy
sample data as input. To use any other stream, the filename / path for the
input file can be specified in the 'h264_video_decode.yaml' file.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded when building the application.

## Building And Running H.264 Endoscopy Tool Tracking Application

Follow steps in README.md from parents directory to build and run the Holohub
dev container. Once inside the Holohub dev container, follow steps mentioned
below to build and run H.264 Endoscopy Tool Tracking application.

## Building the application

Once inside Holohub dev container, run below command from a top level Holohub
directory.

```bash
./run build h264_video_decode
```

## Running the application

* Running the application from the top level Holohub directory

```bash
./run launch h264_video_decode
```

* Running the application `h264_video_decode` from the build directory.

```bash
cd <build_dir>/applications/h264/h264_video_decode/ \
  && ./h264_video_decode --data <HOLOHUB_DATA_DIR>/endoscopy
```
