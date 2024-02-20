### Video Decoder Context

The `video_decoder_context` is used to hold common variables and underlying context. The decoder context handle is passed to both `holoscan::ops::VideoDecoderRequestOp` and `holoscan::ops::VideoDecoderResponseOp`.

#### `holoscan::ops::VideoDecoderContext`

A class used to hold common variables and underlying context required by `holoscan::ops::VideoDecoderRequestOp` and `holoscan::ops::VideoDecoderResponseOp`.

This implementation is based on the GXF Component `nvidia::gxf::VideoDecoderContext`.

##### Parameters

- **`async_scheduling_term`**: Asynchronous scheduling condition required to get/set event state.
  - type: `std::shared_ptr<holoscan::AsynchronousCondition>`

