# DDS Video Publisher Operator

Before using this operator, read the
[RTI Connext DDS Module overview](../../../../modules/holoscan-connext-dds/README.md)
for supported versions, container requirements, and license setup.

`holoscan::ops::DDSVideoPublisherOp` converts each incoming Holoscan video
buffer into a strongly typed DDS [`VideoFrame`](../VideoFrame.idl) sample and
publishes it on the `VideoFrame` topic.

## Input

- **`input`** (`nvidia::gxf::VideoBuffer`): video frame to publish. Only RGBA
  buffers are supported.

## Parameters

- **`writer_qos`** (`std::string`, default: empty): DataWriter QoS profile name
  resolved by the inherited QoS provider.
- **`stream_id`** (`uint32_t`, default: `0`): identifies the published stream.
- The operator also inherits `qos_provider`, `participant_qos`, and `domain_id`
  from [`DDSOperatorBase`](../../base/README.md).

## DDS representation

For every frame, the operator publishes:

- `stream_id`: configured stream identifier.
- `frame_num`: monotonically increasing frame number, starting at zero.
- `width` and `height`: dimensions of the input frame.
- `data`: the complete RGBA frame as a DDS octet sequence.

## Memory and lifecycle behavior

The DDS sample owns a CPU-side copy of every frame. Host buffers are copied
with `memcpy`; device-backed buffers are copied to host memory with
`cudaMemcpy`. This implementation is therefore not zero-copy and is intended
as a straightforward interoperability example.

The DataWriter is created when the operator initializes and is closed during
operator shutdown.

## Limitations

- Only RGBA input is accepted.
- Each published frame requires a full copy into the DDS sample.
- The topic name is fixed to `VideoFrame` by the supplied IDL.
- QoS compatibility and resource limits must be selected for the expected
  resolution, frame rate, and network.

See the [DDS video application](../../../../applications/dds/dds_video/README.md)
for a complete publisher pipeline.
