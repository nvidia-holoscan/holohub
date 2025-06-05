# NVIDIA Video Codec Demo

This application demonstrates the use of NVIDIA Video Codec SDK. The application load a video file, encode the video using either H.264 or HEVC (H.265), decode the video and display it with Holoviz.

### Requirements

- [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)


## Building and Running the NVIDIA Video Codec Application

### Python

```bash
./dev_container build_and_run nvidia_video_codec --language python
```

## Configuration

The application is configured with H.264 codec by default, it may be modified in the [nvidia_video_codec.yaml](./nvidia_video_codec.yaml) file:

```
encoder:
  codec: "H264" # H265 or HEVC
  preset: "P3" # P3, P4, P5, P6, P7
  cuda_device_ordinal: 0
  bitrate: 10000000
  frame_rate: 60
  rate_control_mode: 0 # 0: Constant QP, 1: Variable bitrate, 2: Constant bitrate
  multi_pass_encoding: 1 # 0: Disabled, 1: Quarter resolution, 2: Full resolution
```

## Licensing

By using the NVIDIA Video Codec Demo application and its operators, you agree to the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement). If you disagree with the EULA, please do not run this application.