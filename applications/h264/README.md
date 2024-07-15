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
