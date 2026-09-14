# DDS Video Subscriber Operator

Before using this operator, read the
[RTI Connext DDS Module overview](../../../../modules/holoscan-connext-dds/README.md)
for supported versions, container requirements, and license setup.

`holoscan::ops::DDSVideoSubscriberOp` subscribes to the strongly typed DDS
[`VideoFrame`](../VideoFrame.idl) topic and emits each received frame as a
Holoscan RGBA video buffer.

## Output

- **`output`** (`nvidia::gxf::VideoBuffer`): host-backed RGBA frame created from
  a valid DDS `VideoFrame` sample.

## Parameters

- **`allocator`** (`std::shared_ptr<holoscan::Allocator>`, required): allocator
  used to create output video buffers.
- **`reader_qos`** (`std::string`, default: empty): DataReader QoS profile name
  resolved by the inherited QoS provider.
- **`stream_id`** (`uint32_t`, required): selects the stream to receive.
- The operator also inherits `qos_provider`, `participant_qos`, and `domain_id`
  from [`DDSOperatorBase`](../../base/README.md).

## Stream selection

The operator creates a DDS content-filtered topic with the expression
`stream_id = %0`. DDS delivers only samples whose key matches the configured
`stream_id`, so multiple video streams can share the `VideoFrame` topic.

## Memory and scheduling behavior

The subscriber takes the first valid sample available during each compute
call, allocates a host-backed RGBA buffer, and copies the DDS octet sequence
into it. This implementation is not zero-copy and does not emit device-backed
memory.

When no valid sample is available, the operator returns after a short wait so
the application can respond to shutdown. The DataReader is closed during
operator shutdown.

## Limitations

- Output is always host-backed RGBA memory.
- Each received frame requires a full copy from the DDS sample.
- At most one valid frame is emitted per compute call.
- The topic name is fixed to `VideoFrame` by the supplied IDL.
- The current implementation validates neither the DDS byte-sequence length
  nor additional pixel-format metadata beyond the fixed RGBA contract.

See the [DDS video application](../../../../applications/dds/dds_video/README.md)
for display and headless subscriber pipelines.
