# NVIDIA Video Codec Operators

This directory includes the `nv_video_decoder` and `nv_video_encoder` operators, which are based on the 
[NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk).

These encoder and decoder operators are designed for streaming applications. The encoded frames are stored on the host (CPU) 
memory, where they can be copied to another network streaming operator.

## Sample Application

A sample application can be found in the [nvidia_video_codec](../../applications/nvidia_video_codec/python/README.md) directory.

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by clonging, running, or using the NVIDIA Video Codec sample applications and operators.
