# DDS Video Operators

The DDS video integration consists of two independent C++ operators that use
the shared [`VideoFrame`](VideoFrame.idl) DDS type:

- [`DDSVideoPublisherOp`](dds_video_publisher/README.md) converts RGBA Holoscan
  video buffers into DDS `VideoFrame` samples.
- [`DDSVideoSubscriberOp`](dds_video_subscriber/README.md) receives one DDS
  video stream and emits RGBA Holoscan video buffers.

Each operator README documents its ports, parameters, memory behavior, QoS,
and current limitations. Both operators use
[`DDSOperatorBase`](../base/README.md) for participant and QoS configuration.

See the [DDS video application](../../../applications/dds/dds_video/README.md)
for camera, synthetic-source, display, and headless examples.
