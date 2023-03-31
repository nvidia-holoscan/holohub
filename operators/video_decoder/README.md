### Video Decoder

The `video_decoder` decodes h264 bit stream to YUV.

#### `holoscan::ops::VideoDecoderOp`

Operator class to decode h264 bit stream to YUV.

This implementation is based on `nvidia::gxf::VideoDecoder`.

##### Parameters

- **`image_receiver`**: Receiver to get the input image
  - type: `holoscan::IOSpec*`
- **`output_transmitter`**: Transmitter to send the deocded data
  - type: `holoscan::IOSpec*`
- **`pool`**: Memory pool for allocating output data
  - type: `std::shared_ptr<Allocator>`
- **`inbuf_storage_type`**: Input Buffer storage type, 0:kHost, 1:kDevice
  - type: `int32_t`
- **`outbuf_storage_type`**: Output Buffer storage type, 0:kHost, 1:kDevice
  - type: `int32_t`
- **`device`**: Path to the V4L2 device. Default:"/dev/nvidia0"
  - type: `std::string`
- **`codec`**: Video codec,  0:H264, only H264 supported. Default:0
  - type: `int32_t`
