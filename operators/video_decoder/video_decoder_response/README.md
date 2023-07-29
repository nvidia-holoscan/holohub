### Video Decoder Response

The `video_decoder_response` handles the output of the decoded H264 bit stream.

#### `holoscan::ops::VideoDecoderResponseOp`

Operator class to handle the output of the decoded H264 bit stream.

This implementation is based on `nvidia::gxf::VideoDecoderResponse`.

##### Parameters

- **`output_transmitter`**: Transmitter to send the yuv data.
  - type: `holoscan::IOSpec*`
- **`pool`**: Memory pool for allocating output data.
  - type: `std::shared_ptr<Allocator>`
- **`outbuf_storage_type`**: Output Buffer Storage(memory) type used by this allocator. Can be 0: kHost, 1: kDevice.
  - type: `uint32_t`
- **`videodecoder_context`**: Decoder context Handle.
  - type: `std::shared_ptr<holoscan::ops::VideoDecoderContext>`
