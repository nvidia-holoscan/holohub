# Distributed H.264 Endoscopy Tool Tracking Application

This application is similar to the [H.264 Endoscopy Tool Tracking](../h264_endoscopy_tool_tracking/) application, but this distributed version divides the application into three fragments:

1. Video Input: get video input from a pre-recorded video file.
2. Inference: run the inference using LSTM and run the post-processing script.
3. Visualization: display input video and inference results.


## Requirements

This application is configured to use H.264 elementary stream from endoscopy sample data as input.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded when building the application.

## Building and Running H.264 Endoscopy Tool Tracking Application

* Building and running the application from the top level Holohub directory:

### C++

```bash
# Start the application with all three fragments
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language cpp

# Use the following commands to run the same application three processes:
# Start the application with the video_in fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language cpp --run_args "--driver --worker --fragments video_in --address :10000 --worker-address :10001"
# Start the application with the inference fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language cpp --run_args "--worker --fragments inference --address :10000 --worker-address :10002"
# Start the application with the visualization fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language cpp --run_args "--worker --fragments viz --address :10000 --worker-address :10003"
```
--base_img gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/dev-x86_64:main
### Python

```bash
# Start the application with all three fragments
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language python

# Use the following commands to run the same application three processes:
# Start the application with the video_in fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language python --run_args "--driver --worker --fragments video_in --address :10000 --worker-address :10001"
# Start the application with the inference fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language python --run_args "--worker --fragments inference --address :10000 --worker-address :10002"
# Start the application with the visualization fragment
./dev_container build_and_run h264_endoscopy_tool_tracking_distributed --docker_file applications/h264/Dockerfile --language python --run_args "--worker --fragments viz --address :10000 --worker-address :10003"
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

To start the VS Code Dev Container, run the following command from the root directory of Holohub:

```bash
./dev_container vscode h264
```

### VS Code Launch Profiles

#### C++

Use the **(gdb) h264_endoscopy_tool_tracking_distributed/cpp (all fragments)** launch profile to run and debug the C++ application.

#### Python

Use the **(pythoncpp) h264_endoscopy_tool_tracking_distributed/python (all fragments)** launch profile to run and debug the Python application.
