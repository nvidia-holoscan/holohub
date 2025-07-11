# NVIDIA Video Codec: Video Writer

This application demonstrates the use of NVIDIA Video Codec SDK. The application loads a video file, encodes the video using either H.264 or HEVC (H.265), and then writes the encoded video frames to disk.

### Requirements

- x86 and SBSA platforms with dedicated GPU
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)

> 💡 **Note:** NVIDIA IGX Orin with integrated GPU is not currently supported.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

## Building and Running the NVIDIA Video Codec Application

### Python

```bash
./holohub run nvc_encode_writer --language python
```

## Configuration

The application is configured with H.264 codec by default. It may be modified in the [nvc_encode_writer.yaml](./nvc_encode_writer.yaml) file:

```yaml
encoder:
  codec: "H264" # H265 or HEVC
  preset: "P3" # P1, P2, P3, P4, P5, P6, P7
  cuda_device_ordinal: 0
  bitrate: 10000000
  frame_rate: 60
  rate_control_mode: 0 # 0: Constant QP, 1: Variable bitrate, 2: Constant bitrate
  multi_pass_encoding: 1 # 0: Disabled, 1: Quarter resolution, 2: Full resolution
```

Refer to the [NVIDIA Video Codec documentation](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/) for additional details.

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by cloning, running, or using the NVIDIA Video Codec sample applications and operators.
