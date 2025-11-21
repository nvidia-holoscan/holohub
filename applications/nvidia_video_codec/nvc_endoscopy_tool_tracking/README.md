# NVIDIA Video Codec: Endoscopy Tool Tracking

This application demonstrates GPU-accelerated H.264 video decoding combined with LSTM-based AI inference for endoscopic surgical tool tracking using the NVIDIA Video Codec SDK. The application reads an H.264 elementary stream file, decodes it with hardware acceleration, runs real-time LSTM inference for tool detection and tracking, and optionally encodes the output back to H.264.

This is a modified version of the Endoscopy Tool Tracking reference application in Holoscan SDK that uses NVIDIA Video Codec SDK for efficient video decode and encode operations.

> [!IMPORTANT]
> By using the NVIDIA Video Codec Demo application and its operators, you agree to the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement). If you disagree with the EULA, please do not run this application.

## Key Features

- **Hardware-Accelerated Video Decode**: Uses NVIDIA Video Codec SDK for efficient H.264 decoding
- **AI-Based Tool Tracking**: LSTM neural network for real-time surgical tool detection and tracking
- **Optional Video Encode**: Can encode the processed output back to H.264 format
- **Real-time Visualization**: Displays video with tool tracking overlays using Holoviz
- **Multiple Tool Detection**: Tracks up to 7 different surgical tools (Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, Specimen Bag)

## Requirements

- NVIDIA Driver Version >= 570
- CUDA Version >= 12.8
- x86 and SBSA platforms with dedicated GPU

> 💡 **Note:** NVIDIA IGX Orin with integrated GPU is not currently supported.

## Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded when building the application.

## Building and Running

### Python

```bash
./holohub run nvc_endoscopy_tool_tracking --language python
```

### C++

```bash
./holohub run nvc_endoscopy_tool_tracking --language cpp
```

## Configuration

The application is configured via YAML configuration files:

- **Python**: [python/nvc_endoscopy_tool_tracking.yaml](./python/nvc_endoscopy_tool_tracking.yaml)
- **C++**: [cpp/nvc_endoscopy_tool_tracking.yaml](./cpp/nvc_endoscopy_tool_tracking.yaml)

### Decoder Configuration

```yaml
decoder:
  cuda_device_ordinal: 0
  verbose: false
```

### Encoder Configuration (Optional)

Set `record_output: true` to enable encoding of the processed output:

```yaml
record_output: true  # Setting this to `false` disables H264 encoding

encoder:
  codec: "H264"  # H265 or HEVC
  preset: "P3"   # P1, P2, P3, P4, P5, P6, P7
  cuda_device_ordinal: 0
  bitrate: 10000000
  frame_rate: 60
  rate_control_mode: 1  # 0: Constant QP, 1: Variable bitrate, 2: Constant bitrate
  multi_pass_encoding: 1  # 0: Disabled, 1: Quarter resolution, 2: Full resolution
```

Refer to the [NVIDIA Video Codec documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/) for additional encoder configuration details.

### Input Video File

The application expects H.264 elementary stream files (`.h264` or `.264` extension):

```yaml
reader:
  filename: "surgical_video.264"
  verbose: false
  loop: false
```

## Architecture

The application pipeline consists of:

1. **NvVideoReaderOp**: Reads H.264 elementary stream files
2. **NvVideoDecoderOp**: Hardware-accelerated H.264 decoder
3. **FormatConverterOp**: Converts decoded frames (NV12 → RGB888 → Float32)
4. **LSTMTensorRTInferenceOp**: LSTM-based AI inference for tool tracking
5. **ToolTrackingPostprocessorOp**: Post-processes inference results
6. **HolovizOp**: Visualizes video with tool tracking overlays
7. **NvVideoEncoderOp** (optional): Encodes output to H.264 if `record_output` is enabled

## Preparing H.264 Elementary Stream Files

The application expects H.264 elementary stream files (pure H.264 bitstream without container). You can create these files using FFmpeg:

### From MP4/Container Format

```bash
ffmpeg -i input_video.mp4 -c:v copy -f h264 output.h264
```

### From Endoscopy Sample Data

```bash
ffmpeg -i surgical_video.mp4 -c:v copy -f h264 surgical_video.264
```

## Tool Tracking

The application tracks seven different surgical tools in real-time:

1. **Grasper** - Displayed with red cross overlay
2. **Bipolar** - Forceps for coagulation
3. **Hook** - Electrosurgical hook
4. **Scissors** - Surgical scissors
5. **Clipper** - Clip applier
6. **Irrigator** - Irrigation/suction device
7. **Spec.Bag** - Specimen retrieval bag

Each detected tool is marked with:
- Cross overlay showing tool position
- Text label identifying the tool type
- Colored mask indicating tool region

## Performance Considerations

_The H.264 video decode operators do not adjust framerate as they read the elementary stream input. As a result, the video stream can be displayed as quickly as the decoding and inference can be performed. This application uses `PeriodicCondition` to play video at the same speed as the source video._

The LSTM inference runs at each frame, providing real-time tool tracking with minimal latency.

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by cloning, running, or using the NVIDIA Video Codec sample applications and operators.
