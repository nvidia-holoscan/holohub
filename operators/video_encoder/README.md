### Video Encoder

The `video_encoder` encodes YUV video frames to h264 bit stream.

#### `holoscan::ops::VideoEncoderOp`

Operator class to encode YUV video frames to h264 bit stream.

This implementation is based on `nvidia::gxf::VideoEncoder`.

##### Parameters

- **`input_frame`**: Receiver to get the input frame
  - type: `holoscan::IOSpec*`
- **`output_transmitter`**: Transmitter to send the compressed data
  - type: `holoscan::IOSpec*`
- **`pool`**: Memory pool for allocating output data
  - type: `std::shared_ptr<Allocator>`
- **`inbuf_storage_type`**: Input Buffer storage type, 0:kHost, 1:kDevice
  - type: `int32_t`
- **`outbuf_storage_type`**: Output Buffer storage type, 0:kHost, 1:kDevice
  - type: `int32_t`
- **`device`**: Path to the V4L2 device. Default:"/dev/nvidia0"
  - type: `std::string`
- **`codec`**: Video Codec to use,  0:H264, only H264 supported. Default:0
  - type: `int32_t`
- **`input_height`**: Input frame height
  - type: `uint32_t`
- **`input_width`**: Input image width
  - type: `uint32_t`
- **`input_format`**: Input frame color format, nv12 PL is supported. Default:"nv12pl"
  - type: `std::string`
- **`profile`**: Encode profile, 0:Baseline Profile, 1: Main , 2: High
  - type: `int32_t`
- **`bitrate`**: Encoder bitrate, Bitrate of the encoded stream, in bits per second. Default:20000000
  - type: `int32_t`
- **`framerate`**: Frame Rate, FPS. Default:30
  - type: `int32_t`
