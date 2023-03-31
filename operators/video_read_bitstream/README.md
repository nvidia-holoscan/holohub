### Video Bit Stream Reader

The `video_read_bitstream` reads h264 bit stream from specified input file.

#### `holoscan::ops::VideoReadBitstreamOp`

Operator class to read H264 video bit stream.

This implementation is based on `nvidia::gxf::VideoReadBitStream`.

##### Parameters

- **`output_transmitter`**: Transmitter to send the compressed data
  - type: `holoscan::IOSpec*`
- **`input_file_path`**: Path to image file
  - type: `std::string`
- **`pool`**: Memory pool for allocating output data
  - type: `std::shared_ptr<Allocator>`
- **`outbuf_storage_type`**: Output Buffer storage type, 0:kHost, 1:kDevice
  - type: `int32_t`
