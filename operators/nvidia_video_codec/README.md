# NVIDIA Video Codec Operators

This directory includes the `nv_video_decoder`, `nv_video_encoder`, and `nv_video_reader` operators, which are based on the  
[NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk).

These encoder and decoder operators are designed for streaming applications. The encoded frames are stored on the host (CPU) 
memory, where they can be copied to another network streaming operator.

> [!IMPORTANT]  
> By using the NVIDIA Video Codec operators, you agree to the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement). If you disagree with the EULA, please do not run this application.

## Sample Applications

- [H.264 File Decoder](../../applications/nvidia_video_codec/nvc_decode/)
- [Encode and Decode](../../applications/nvidia_video_codec/nvc_encode_decode/)
- [Video Writer](../../applications/nvidia_video_codec/nvc_encode_writer/)

## Licensing

Holohub applications and operators are licensed under Apache-2.0.

NVIDIA Video Codec is governed by the terms of the [NVIDIA Software Developer License Agreement](https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement), which you accept by cloning, running, or using the NVIDIA Video Codec sample applications and operators.
