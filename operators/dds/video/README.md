### DDS Video Operators

The DDS Video Operators allow applications to read or write video buffers
to a DDS databus, enabling communication with other applications via the
[VideoFrame](VideoFrame.idl) DDS topic.

#### `holoscan::ops::DDSVideoPublisherOp`

Operator class for the DDS video publisher. This operator accepts `VideoBuffer` objects
as input and publishes each buffer to DDS as a [VideoFrame](VideoFrame.idl).

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`writer_qos`**: The name of the QoS profile to use for the DDS DataWriter
  - type: `std::string`
- **`stream_id`**: The ID to use for the video stream
  - type: `uint32_t`

##### Inputs

- **`input`**: Input video buffer
  - type: `nvidia::gxf::VideoBuffer`

#### `holoscan::ops::DDSVideoSubscriberOp`

Operator class for the DDS video subscriber. This operator reads from the
[VideoFrame](VideoFrame.idl) DDS topic and outputs each received frame as
`VideoBuffer` objects.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`reader_qos`**: The name of the QoS profile to use for the DDS DataReader
  - type: `std::string`
- **`stream_id`**: The ID of the video stream to filter for
  - type: `uint32_t`
- **`allocator`**: Allocator used to allocate the output data
  - type: `std::shared_ptr<Allocator>`

##### Outputs

- **`output`**: Output video buffer
  - type: `nvidia::gxf::VideoBuffer`
