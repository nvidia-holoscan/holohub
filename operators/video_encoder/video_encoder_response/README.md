### Video Encoder Response

The `video_encoder_response` handles the output of the encoded YUV frames.

#### `holoscan::ops::VideoEncoderResponseOp`

Operator class to handle the output of the encoded YUV frames.

This implementation is based on `nvidia::gxf::VideoEncoderResponse`.

##### Parameters

- **`output_transmitter`**: Transmitter to send the compressed data.
  - type: `holoscan::IOSpec*`
- **`pool`**: Memory pool for allocating output data.
  - type: `std::shared_ptr<Allocator>`
- **`videoencoder_context`**: Encoder context Handle.
  - type: `std::shared_ptr<holoscan::ops::VideoEncoderContext>`
- **`outbuf_storage_type`**: Output Buffer Storage(memory) type used by this allocator. Can be 0: kHost, 1: kDevice. Default: 1.
  - type: `uint32_t`

