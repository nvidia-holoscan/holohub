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

## Building and Running H.264 Endoscopy Tool Tracking Application

* Building and running the application from the top level Holohub directory:

```bash
# C++ version
./dev_container build_and_run h264_video_decode --docker_file applications/h264/Dockerfile --language cpp

# Python version
./dev_container build_and_run h264_video_decode --docker_file applications/h264/Dockerfile --language python

```

Important: on aarch64, applications also need tegra folder mounted inside the container and
the `LD_LIBRARY_PATH` environment variable should be updated to include
tegra folder path.

Open and edit the [Dockerfile](../Dockerfile) and uncomment line 66:

```bash
# Uncomment the following line for aarch64 support
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/
```


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./dev_container vscode h264
```

### VS Code Launch Profiles

#### C++

Use the **(gdb) h264_video_decode/cpp** launch profile to run and debug the C++ application.

#### Python

There are a couple of launch profiles configured for this application:

1. **(debugpy) h264_video_decode/python**: Launch the h.264 Video Decode application with the ability to debug Python code.
2. **(pythoncpp) h264_video_decode/python**: Launch the h.264 Video Decode application with the ability to debug both Python and C++ code.
