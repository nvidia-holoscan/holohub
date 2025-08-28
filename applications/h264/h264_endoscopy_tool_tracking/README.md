# H.264 Endoscopy Tool Tracking

The application showcases how to use H.264 video source as input to and output
from the Holoscan pipeline. This application is a modified version of Endoscopy
Tool Tracking reference application in Holoscan SDK that supports H.264
elementary streams as the input and output.

_The H.264 video decode operators do not adjust framerate as it reads the
elementary stream input. As a result the video stream can be displayed as
quickly as the decoding can be performed. This application uses
`PeriodicCondition` to play video at the same speed as the source video._

## Requirements

This application is configured to use H.264 elementary stream from endoscopy
sample data as input. The recording of the output can be enabled by setting
`record_output` flag in the config file to `true`. If the `record_output` flag
in the config file is set to `true`, the output of the pipeline is again
recorded to a H.264 elementary stream on the disk, file name / path for this
can be specified in the 'h264_endoscopy_tool_tracking.yaml' file.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded when building the application.

## Build and Run the H.264 Endoscopy Tool Tracking Application

### C++

```bash
./holohub run h264_endoscopy_tool_tracking --language cpp
```

### Python

Separate build and run commands are required to address the known [symbol loading issue](../README.md#symbol-error-at-load).

```bash
# C++ version
./holohub build h264_endoscopy_tool_tracking --language python

# Python version
# Note: LD_PRELOAD required to address symbol issue
./holohub run h264_endoscopy_tool_tracking --language python \
    --docker-opts="-e LD_PRELOAD=/opt/nvidia/holoscan/lib/libgxf_core.so"
```

Important: on aarch64, applications also need tegra folder mounted inside the container and
the `LD_LIBRARY_PATH` environment variable should be updated to include
tegra folder path.

Open and edit the [Dockerfile](../Dockerfile) and uncomment line 66:

```bash
# Uncomment the following line for aarch64 support
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/
```


## Enable recording of the output

The recording of the output can be enabled by setting `record_output` flag in
the config file
`<build_dir>/applications/h264/endoscopy_tool_tracking/h264_endoscopy_tool_tracking.yaml`
to `true`.


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./holohub vscode h264
```

### VS Code Launch Profiles

#### C++

Use the **(gdb) h264_endoscopy_tool_tracking/cpp** launch profile to run and debug the C++ application.

#### Python

There are a couple of launch profiles configured for this application:

1. **(debugpy) h264_endoscopy_tool_tracking/python**: Launch the h.264 Endoscopy Tool Tracking application with the ability to debug Python code.
2. **(pythoncpp) h264_endoscopy_tool_tracking/python**: Launch the h.264 Endoscopy Tool Tracking application with the ability to debug both Python and C++ code.
