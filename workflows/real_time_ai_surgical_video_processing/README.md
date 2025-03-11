# Real-Time AI Surgical Video Processing Workflow

## Overview

Holoscan Workflows are end-to-end reference applications that may include Holoscan Sensor Bridge.
In this workflow, we demonstrate a comprehensive real-time AI surgical video processing pipeline that includes:

1. Integration with [Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/introduction.html), which enables an end-to-end pipeline.
2. Out-of-body detection to determine if the endoscope is inside or outside the patient's body.
3. Dynamic flow condition based on out-of-body detection results:
   - Deidentification (pixelation) when outside the body
   - Multi-AI processing when inside the body
4. Multi-AI Surgical Tool processing with:
   - SSD detection for surgical tool detection
   - MONAI segmentation for endoscopic tool segmentation

![Sample Output Images](images/RAISVP-sample-images.png)

Fig.1: Endoscopy image from a partial nephrectomy procedures (surgical removal of the diseased portion of the kidney) showing AI tool segmentation results when the camera is inside the body and a deidenfied (pixelated) output image when the camera is outside of the body.
Image courtesy of Research Group Camma, IHU Strasbourg and the University of Strasbourg ([NGC Resource](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data))

## Workflow Components
![RAISVP-workflow](./images/RAISVP-dynamic-workflow.png)
Fig.2: The workflow diagram representing all the holoscan operators (in green) and holoscan sensor bridge operators (in yellow). The source can be a Holoscan Sensor Bridge, an AJA Card and a video replayer.

### 1. Out-of-Body Detection

The workflow first determines if the endoscope is inside or outside the patient's body using an AI model.

### 2. Conditional Processing

- If outside the body: The video is deidentified through pixelation to protect privacy
- If inside the body: The video is processed by the multi-AI pipeline

### 3. Multi-AI Processing

When inside the body, two AI models run concurrently:

- SSD detection model identifies surgical tools with bounding boxes
- MONAI segmentation model provides pixel-level segmentation of tools

### 4. Visualization

The HolovizOp displays the processed video with overlaid AI results, including:

- Bounding boxes around detected tools
- Segmentation masks for tools
- Text labels for detected tools

## Models

This workflow utilizes three AI models:

- [Out-of-body Detection Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_orsi_academy_sample_data): `anonymization_model.onnx`
- [SSD Detection for Endoscopy Surgical Tools](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/ssd_surgical_tool_detection_model) with additional NMS op: `epoch24_nms.onnx`
- [MONAI Endoscopic Tool Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/monai_endoscopic_tool_segmentation_model): `model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx`

## Data

[📦️ (NGC) Orsi partial nephrectomy procedures](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

## Requirements

Ensure you have installed the Holoscan SDK via one of the methods specified in [the SDK user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#development-software-stack).

The directory specified by `--data` at app runtime is assumed to contain three subdirectories, corresponding to the NGC resources specified in [Model](#models) and [Data](#data): `endoscopy`, `endoscopy_out_of_body_detection`, `monai_tool_seg_model` and `ssd_model`. These resources will be automatically downloaded to the Holohub data directory when building the application.

## Building the docker image for workflow (using Holoscan Sensor Bridge)

The workflow requires the use of the Holoscan Sensor Bridge. Thus you need a Holoscan Sensor Bridge container, which can be built using the following command:

```sh
git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
cd holoscan-sensor-bridge
./docker/build.sh
```

This will build a docker image called `hololink-demo:{HSB_VERSION}`
Once you have built the Holoscan Sensor Bridge container, you can build the Holohub container using the following command:

```sh
./dev_container build --base_img hololink-demo:2.0.0 --img holohub:link
```

## Building the workflow

First you need to run the Holohub container:

```sh
./dev_container launch --img holohub:link 
```

Then, you can build the workflow using the following command:

```sh
./run build real_time_ai_surgical_video_processing
```

## Running the application

### Use Holohub container from outside of the container

Using the Holohub container, you can run the workflow without building it again:

```sh
./dev_container build_and_run --base_img hololink-demo:2.0.0 --img holohub:link --no_build real_time_ai_surgical_video_processing
```

However, if you want to build the workflow, you can just remove the `--no_build` flag:

```sh
./dev_container build_and_run --base_img hololink-demo:2.0.0 --img holohub:link real_time_ai_surgical_video_processing
```

### Use Holohub container from inside the container

First you need to run the Holohub container:

```sh
./dev_container launch --img holohub:link 
```

To run the Python application, you can make use of the run script

```sh
./run launch real_time_ai_surgical_video_processing python
```

Alternatively, you can run the application directly:

```sh
cd <HOLOHUB_SOURCE_DIR>/workflows/real_time_ai_surgical_video_processing/python
python3 real_time_ai_surgical_video_processing.py --source hsb --data <DATA_DIR> --config <CONFIG_FILE>
```

### Command Line Arguments

The application accepts the following command line arguments:

- `-s, --source`: Source of video input. Options are:
  - `replayer`: Use prerecorded video from the endoscopy dataset
  - `aja`: Use an AJA capture card as the source
  - `hsb`: Use the Holoscan Sensor Bridge as the source
  Default: `replayer`

- `-c, --config`: Path to a custom configuration file
  Default: `config.yaml` in the application directory

- `-d, --data`: Path to the data directory containing model and video files
  Default: Uses the HOLOHUB_DATA_PATH environment variable

## Workflow Components

### 1. Out-of-Body Detection

The workflow first determines if the endoscope is inside or outside the patient's body using an AI model.

### 2. Conditional Processing

- If outside the body: The video is deidentified through pixelation to protect privacy
- If inside the body: The video is processed by the multi-AI pipeline

### 3. Multi-AI Processing

When inside the body, two AI models run concurrently:

- SSD detection model identifies surgical tools with bounding boxes
- MONAI segmentation model provides pixel-level segmentation of tools

### 4. Visualization

The HolovizOp displays the processed video with overlaid AI results, including:

- Bounding boxes around detected tools
- Segmentation masks for tools
- Text labels for detected tools

## Benchmarking
![Latency Profiling](./images/RAISVP_latency.png)