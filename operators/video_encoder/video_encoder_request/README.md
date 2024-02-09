### Video Encoder Request

The `video_encoder_request` handles the input for encoding YUV frames to H264 bit stream.

#### `holoscan::ops::VideoEncoderOp`

Operator class to handle the input for encoding YUV frames to H264 bit stream.

This implementation is based on `nvidia::gxf::VideoEncoderRequest`.

##### Parameters

- **`input_frame`**: Receiver to get the input frame.
  - type: `holoscan::IOSpec*`
- **`videoencoder_context`**: Encoder context Handle.
  - type: `std::shared_ptr<holoscan::ops::VideoEncoderContext>`
- **`inbuf_storage_type`**: Input Buffer storage type, 0: kHost, 1: kDevice. Default: 1
  - type: `uint32_t`
- **`codec`**: Video codec to use,  0: H264, only H264 supported. Default: 0.
  - type: `int32_t`
- **`input_height`**: Input frame height.
  - type: `uint32_t`
- **`input_width`**: Input image width.
  - type: `uint32_t`
- **`input_format`**: Input color format, nv12,nv24,yuv420planar. Default: nv12.
  - type: `nvidia::gxf::EncoderInputFormat`
- **`profile`**: Encode profile, 0: Baseline Profile, 1: Main, 2: High. Default: 2.
  - type: `int32_t`
- **`bitrate`**: Bitrate of the encoded stream, in bits per second. Default: 20000000.
  - type: `int32_t`
- **`framerate`**: Frame Rate, frames per second. Default: 30.
  - type: `int32_t`
- **`qp`**: Encoder constant QP value. Default: 20.
  - type: `uint32_t`
- **`level`**: Video H264 level. Maximum data rate and resolution, select from 0 to 14. Default: 14.
  - type: `int32_t`
- **`iframe_interval`**: I Frame Interval, interval between two I frames. Default: 30.
  - type: `int32_t`
- **`rate_control_mode`**: Rate control mode, 0: CQP[RC off], 1: CBR, 2: VBR. Default: 1.
  - type: `int32_t`
- **`config`**: Preset of parameters, select from pframe_cqp, iframe_cqp, custom. Default: custom.
  - type: `nvidia::gxf::EncoderConfig`

