# DDS Video Operators

The DDS Video Operators allow applications to read or write video buffers
to a DDS databus, enabling communication with other applications via the
[VideoFrame](VideoFrame.idl) DDS topic.

This operator requires an installation of [RTI Connext](https://content.rti.com/l/983311/2025-07-08/q5x1n8) to provide access to the DDS domain, as specified by the [OMG Data-Distribution Service](https://www.omg.org/omg-dds-portal/).

You can obtain a license/activation key for RTI Connext directly from RTI by downloading it [here](https://content.rti.com/l/983311/2025-07-25/q6729c). For additional information on RTI Connext and how it integrates with NVIDIA products, please refer to the [RTI-NVIDIA integration page](https://www.rti.com/products/third-party-integrations/nvidia).

If you have questions, please email [evaluations@rti.com](mailto:evaluations@rti.com).

## `holoscan::ops::DDSVideoPublisherOp`

Operator class for the DDS video publisher. This operator accepts `VideoBuffer` objects
as input and publishes each buffer to DDS as a [VideoFrame](VideoFrame.idl).

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

### Parameters

- **`writer_qos`**: The name of the QoS profile to use for the DDS DataWriter
  - type: `std::string`
- **`stream_id`**: The ID to use for the video stream
  - type: `uint32_t`

### Inputs

- **`input`**: Input video buffer
  - type: `nvidia::gxf::VideoBuffer`

## `holoscan::ops::DDSVideoSubscriberOp`

Operator class for the DDS video subscriber. This operator reads from the
[VideoFrame](VideoFrame.idl) DDS topic and outputs each received frame as
`VideoBuffer` objects.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

### Parameters

- **`reader_qos`**: The name of the QoS profile to use for the DDS DataReader
  - type: `std::string`
- **`stream_id`**: The ID of the video stream to filter for
  - type: `uint32_t`
- **`allocator`**: Allocator used to allocate the output data
  - type: `std::shared_ptr<Allocator>`

### Outputs

- **`output`**: Output video buffer
  - type: `nvidia::gxf::VideoBuffer`
