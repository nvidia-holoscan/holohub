### Video Encoder Context

The `video_encoder_context` is used to hold common variables and underlying context. The encoder context handle is passed to both `holoscan::ops::VideoEncoderRequestOp` and `holoscan::ops::VideoEncoderResponseOp`.

#### `holoscan::ops::VideoEncoderContext`

A class used to hold common variables and underlying context required by `holoscan::ops::VideoEncoderRequestOp` and `holoscan::ops::VideoEncoderResponseOp`.

This implementation is based on the GXF Component `nvidia::gxf::VideoEncoderContext`.
