### Video Decoder Request

The `video_decoder_request` handles the input for the H264 bit stream decode.

#### `holoscan::ops::VideoDecoderRequestOp`

Operator class to handle the input for the H264 bit stream decode.

This implementation is based on `nvidia::gxf::VideoDecoderRequest`.

##### Parameters

- **`input_frame`**: Receiver to get the input image.
  - type: `holoscan::IOSpec*`
- **`inbuf_storage_type`**: Input Buffer storage type, 0:kHost, 1:kDevice.
  - type: `uint32_t`
- **`async_scheduling_term`**: Asynchronous scheduling condition.
  - type: `std::shared_ptr<holoscan::AsynchronousCondition>`
- **`videodecoder_context`**: Decoder context Handle.
  - type: `std::shared_ptr<holoscan::ops::VideoDecoderContext>`
- **`codec`**: Video codec to use,  0:H264, only H264 supported. Default:0.
  - type: `uint32_t`
- **`disableDPB`**: Enable low latency decode, works only for IPPP case.
  - type: `uint32_t`
- **`output_format`**: Output frame video format, nv12pl and yuv420planar are supported.
  - type: `std::string`
