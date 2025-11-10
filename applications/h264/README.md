# H.264 Encode / Decode Reference Applications

This folder contains two reference applications that showcases the use of H.264
Encode / Decode operators to read, decode H.264 elementary streams and encode,
write H.264 elementary streams to the disk.

## H.264 Endoscopy Tool Tracking Application

The application showcases how to use H.264 video source as input to and output
from the Holoscan pipeline. This application is a modified version of Endoscopy
Tool Tracking reference application in Holoscan SDK that supports H.264
elementary streams as the input and output.

[Building and Running the H.264 Endoscopy Tool Tracking Application](./h264_endoscopy_tool_tracking/README.md)

## H.264 Video Decode Application

This is a minimal reference application demonstrating usage of H.264 video
decode operators. This application makes use of H.264 elementary stream reader
operator for reading H.264 elementary stream input and uses Holoviz operator
for rendering decoded data to the native window.

[Building and Running the H.264 Video Decode Application](./h264_video_decode//README.md)

## Supported Platforms

- x86_64
- arm64 + discrete GPU platforms (SBSA)

> [!IMPORTANT]  
> Starting from Holoscan 3.6.0, H.264 applications only support CUDA 13 or higher.
> To run H.264 application with CUDA 12, please use tags with [holoscan-sdk-3.5.0](https://github.com/nvidia-holoscan/holohub/tree/holoscan-sdk-3.5.0) or earlier.

## Known Issues

### Unsupported Platforms

Integrated GPU devices in the Orin family (Jetson AGX Orin, IGX Orin) are not supported.

Platforms with NVIDIA drivers < 580.00 are not supported.

### Symbol error at load

Python applications have been observed to emit the following at runtime:
```bash
2: [warning] [gxf_extension_manager.cpp:174] Unable to load extension from 'libgxf_videodecoder.so' \
  (error: /opt/nvidia/holoscan/lib/libgxf_videodecoder.so: undefined symbol: _ZN6nvidia6logger15GlobalGxfLogger8instanceEv)
2: [warning] [gxf_extension_manager.cpp:174] Unable to load extension from 'libgxf_videodecoderio.so' \
  (error: /opt/nvidia/holoscan/lib/libgxf_videodecoderio.so: undefined symbol: _ZN6nvidia6logger15GlobalGxfLogger8instanceEv)
2: [info] [fragment.cpp:778] Loading extensions from configs...
2: [warning] [type_registry.cpp:57] Unknown type: nvidia::gxf::VideoReadBitStream
```

Please explicitly pre-load the GXF Core library to resolve the error.

```bash
LD_PRELOAD=/opt/nvidia/holoscan/lib/libgxf_core.so <cmd> ...
```
