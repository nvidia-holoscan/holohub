# Real-Time End-to-end AI Surgical Video Workflow

![Overall Diagram](images/RAISVP_overall_diagram.png)
*Fig.1: The overall diagram illustrating the end-to-end pipeline for real-time AI surgical video processing. The pipeline achieves an average end-to-end latency of 37ms (maximum 54ms). Key latency components are shown: Holoscan Sensor Bridge (HSB) latency averages 21ms (max 28ms), and the AI application averages 16ms (median 17ms, 95th percentile 18ms, 99th percentile 22ms, max 26ms). These results demonstrate the solution's high-performance, low-latency capabilities for demanding surgical video applications.*

## Overview

This reference application offers developers a modular, end-to-end pipeline that spans the entire sensor processing workflow—from sensor data ingestion and accelerated computing to AI inference, real-time visualization, and data stream output.

Specifically, we demonstrate a comprehensive real-time end-to-end AI surgical video pipeline that includes:

1. **Sensor I/O**: Integration with [Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/introduction.html), enabling GPU Direct data ingestion for ultra low-latency input of surgical video feeds.
2. **Out-of-body detection** to determine if the endoscope is inside or outside the patient's body, ensuring patient privacy by removing identifiable information.
3. **Dynamic flow condition** based on out-of-body detection results.
4. **De-identification**: pixelate the image to anonymize outside of body elements like people's faces.
5. **Multi-AI**: Enabling simultaneous execution of multiple models at inference. Surgical Tool processing with:
   - SSD detection for surgical tool detection
   - MONAI segmentation for endoscopic tool segmentation

## Architecture

![RAISVP-workflow](./images/RAISVP_dynamic_workflow.png)
*Fig.2: The workflow diagram representing all the holoscan operators (in green) and holoscan sensor bridge operators (in yellow). The source can be a Holoscan Sensor Bridge, an AJA Card or a video replayer.*

![Sample Output Images](images/RAISVP_sample_images.png)
*Fig.3: Endoscopy image from a partial nephrectomy procedure (surgical removal of the diseased portion of the kidney) showing AI tool segmentation results when the camera is inside the body and a deidentified (pixelated) output image when the camera is outside of the body.*

### 1. Out-of-Body Detection

The workflow first determines if the endoscope is inside or outside the patient's body using an AI model.

### 2. Dynamic Flow Control

- **If outside the body**: The video is deidentified through pixelation to protect privacy
- **If inside the body**: The video is processed by the multi-AI pipeline

### 3. Multi-AI Processing

When inside the body, two AI models run concurrently:

- **SSD detection model** identifies surgical tools with bounding boxes
- **MONAI segmentation model** provides pixel-level segmentation of tools

### 4. Visualization

The HolovizOp displays the processed video with overlaid AI results, including:

- Bounding boxes around detected tools
- Segmentation masks for tools
- Text labels for detected tools

## Requirements

### Software

- **Holoscan SDK `v3.0`**:
  Holohub command takes care of this dependency when using Holohub container. However, you can install the Holoscan SDK via one of the methods specified in [the SDK user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#development-software-stack).
- **Holoscan Sensor Bridge `v2.0`**: Please see the [Quick start guide](#quick-start-guide) for building the Holoscan Sensor Bridge docker container.

### Models

This workflow utilizes the following three AI models:

| Model | Description | File |
|-------|-------------|------|
| [📦️ Out-of-body Detection Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_orsi_academy_sample_data) | Detects if endoscope is inside or outside the body | `anonymization_model.onnx` |
| [📦️ SSD Detection for Endoscopy Surgical Tools](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/ssd_surgical_tool_detection_model) | Detects surgical tools with bounding boxes | `epoch24_nms.onnx` |
| [📦️ MONAI Endoscopic Tool Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/monai_endoscopic_tool_segmentation_model) | Provides pixel-level segmentation of tools | `model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx` |

### Sample Data

- [📦️ Orsi partial nephrectomy procedures](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data) - Sample endoscopy video data for use with the `replayer` source
  
> **Note:** The directory specified by `--data` at runtime is assumed to contain three subdirectories, corresponding to the NGC resources specified in [Models](#models) and [Sample Data](#sample-data): `orsi`, `monai_tool_seg_model` and `ssd_model`. These resources will be automatically downloaded to the Holohub data directory when building the application.

## Quick Start Guide

### Using AJA Card or Replayer as I/O

```sh
./dev_container build_and_run ai_surgical_video
```

### Using Holoscan Sensor Bridge as I/O

When using the workflow with `--source hsb`, it requires the Holoscan Sensor Bridge software to be installed. You can build a Holoscan Sensor Bridge container using the following commands:

```sh
git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
cd holoscan-sensor-bridge
git checkout hsdk-3.0
./docker/build.sh
```

This will build a docker image called `hololink-demo:2.0.0`.

Once you have built the Holoscan Sensor Bridge container, you can build the Holohub container and run the workflow using the following command:

```sh
./dev_container build_and_run --base_img hololink-demo:2.0.0 --img holohub:link ai_surgical_video --run_args " --source hsb"
```

## Advanced Usage

### Building the Application

First, you need to run the Holohub container:

```sh
./dev_container launch --img holohub:link 
```

Then, you can build the workflow using the following command:

```sh
./run build ai_surgical_video
```

### Running the Application

#### Use Holohub Container from Outside of the Container

Using the Holohub container, you can run the workflow without building it again:

```sh
./dev_container build_and_run --base_img hololink-demo:2.0.0 --img holohub:link --no_build ai_surgical_video
```

However, if you want to build the workflow, you can just remove the `--no_build` flag:

```sh
./dev_container build_and_run --base_img hololink-demo:2.0.0 --img holohub:link ai_surgical_video
```

#### Use Holohub Container from Inside the Container

First, you need to run the Holohub container:

```sh
./dev_container launch --img holohub:link 
```

To run the Python application, you can make use of the run script:

```sh
./run launch ai_surgical_video
```

Alternatively, you can run the application directly:

```sh
cd <HOLOHUB_SOURCE_DIR>/workflows/ai_surgical_video/python
python3 ai_surgical_video.py --source hsb --data <DATA_DIR> --config <CONFIG_FILE>
```

> **TIP:** You can get the exact "Run command" along with "Run environment" and "Run workdir" by executing:
>
> ```bash
> ./run launch ai_surgical_video --dryrun
> ```

### Command Line Arguments

The application accepts the following command line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `-s, --source` | Source of video input: `replayer`, `aja`, or `hsb` | `replayer` |
| `-c, --config` | Path to a custom configuration file | `config.yaml` in the application directory |
| `-d, --data` | Path to the data directory containing model and video files | Uses the `HOLOHUB_DATA_PATH` environment variable |
| `--headless` | Run in headless mode (no visualization) | False |
| `--fullscreen` | Run in fullscreen mode | False |
| `--camera-mode` | Camera mode to use [0,1,2,3] | `0` |
| `--frame-limit` | Exit after receiving this many frames | No limit |
| `--hololink` | IP address of Hololink board | `192.168.0.2` |
| `--log-level` | Logging level to display | `20` |
| `--ibv-name` | IBV device to use | First available InfiniBand device |
| `--ibv-port` | Port number of IBV device | `1` |
| `--expander-configuration` | I2C Expander configuration (0 or 1) | `0` |
| `--pattern` | Configure to display a test pattern (0-11) | None |
| `--ptp-sync` | After reset, wait for PTP time to synchronize | False |
| `--skip-reset` | Don't call reset on the hololink device | False |

## Benchmarking

Please refer to [Holoscan Benchmarking](../../benchmarks/holoscan_flow_benchmarking/README.md) for how to perform benchmarking for this workflow.
