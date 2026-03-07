# NVIDIA Video Codec: Encode-Decode Video

This application demonstrates the use of NVIDIA Video Codec SDK. The application loads a video file, encodes the video using either H.264 or HEVC (H.265), decodes the video, and displays it with Holoviz.

> [!IMPORTANT]
> By using the NVIDIA Video Codec Demo application and its operators, you agree to the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement). If you disagree with the EULA, please do not run this application.

## Requirements

- NVIDIA Driver Version >= 570
- CUDA Version >= 12.8
- x86 and SBSA platforms with dedicated GPU

> 💡 **Note:** NVIDIA IGX Orin with integrated GPU is not currently supported.

## Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

## Building and Running the NVIDIA Video Codec Application

### Python

```bash
./holohub run nvc_encode_decode --language python
```

### C++

```bash
./holohub run nvc_encode_decode --language cpp
```

## Configuration

The application is configured with H.264 codec by default. It may be modified in the configuration files:

- **Python**: [python/nvc_encode_decode.yaml](./python/nvc_encode_decode.yaml)
- **C++**: [cpp/nvc_encode_decode.yaml](./cpp/nvc_encode_decode.yaml)

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

## Benchmarks

We collected latency benchmark results using Holoscan [Data Flow Tracking](https://docs.nvidia.com/holoscan/sdk-user-guide/flow_tracking.html) tools on the NVIDIA Video Codec sample application. The benchmark is conducted on x86_64 with AMD Ryzen 9 7950X, 128 GB system memory and NVIDIA ADA6000 GPU.

**Encoder Configurations:**

- **Bitrate**: 10 Mbps
- **FPS**: 60
- **Rate Control Mode**: 1 Variable Bitrate
- **Multi-pass Encoding**: 1 Quarter Resolution

| Codec | Preset | E2E Min | E2E Max | E2E Avg | Enc Min | Enc Max | Enc Avg | Dec Min | Dec Max | Dec Avg | FPS Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H.264 | P3 | 6.242 | 8.423 | 6.743 | 0.536 | 0.865 | 0.593 | 5.258 | 7.307 | 5.699 | 145.270 |
| H.264 | P4 | 6.22 | 8.219 | 6.674 | 0.561 | 0.962 | 0.615 | 5.220 | 7.218 | 5.616 | 146.875 |
| H.264 | P5 | 6.508 | 8.441 | 7.044 | 0.921 | 1.403 | 0.971 | 5.229 | 7.097 | 5.658 | 139.433 |
| H.264 | P6 | 6.37 | 9.409 | 7.102 | 0.680 | 1.060 | 0.730 | 5.141 | 7.301 | 5.646 | 143.368 |
| H.264 | P7 | 6.529 | 8.531 | 7.107 | 0.740 | 1.155 | 0.795 | 5.104 | 8.231 | 5.650 | 142.727 |
| HEVC | P3 | 6.258 | 9.039 | 6.898 | 0.684 | 1.078 | 0.728 | 5.141 | 7.371 | 5.656 | 145.058 |
| HEVC | P4 | 6.146 | 9.351 | 6.788 | 0.684 | 1.088 | 0.731 | 5.130 | 7.301 | 5.642 | 143.481 |
| HEVC | P5 | 6.193 | 8.991 | 6.818 | 0.680 | 1.060 | 0.730 | 5.130 | 7.301 | 5.643 | 143.371 |
| HEVC | P6 | 6.337 | 9.113 | 6.924 | 0.682 | 1.109 | 0.733 | 5.254 | 8.043 | 5.729 | 141.633 |
| HEVC | P7 | 6.246 | 10.11 | 6.909 | 0.740 | 1.155 | 0.793 | 5.104 | 8.231 | 5.667 | 142.035 |

*Note: all reported latency values are in milliseconds.*

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by cloning, running, or using the NVIDIA Video Codec sample applications and operators.
