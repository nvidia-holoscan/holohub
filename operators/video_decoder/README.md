### Video Decoder Operators

The `video_decoder` is a set of operators that can be used together to decode h264 bit stream to YUV.

#### `holoscan::ops::VideoDecoderRequestOp`

Operator class to handle the decoder input.

This implementation is based on `nvidia::gxf::VideoDecoderRequest`.

#### `holoscan::ops::VideoDecoderResponseOp`

Operator class to handle the decoder output.

This implementation is based on `nvidia::gxf::VideoDecoderResponse`.

#### `holoscan::ops::VideoDecoderContext`

`VideoDecoderContext` is used to hold common variables and underlying context. The decoder context handle is passed to both `VideoDecoderRequestOp` and `VideoDecoderRequestOp`

This implementation is based on `nvidia::gxf::VideoDecoderContext`.
