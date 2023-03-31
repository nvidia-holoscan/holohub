### Video Bit Stream Writer

The `video_write_bitstream` writes bit stream to the disk at specified output
path.

#### `holoscan::ops::VideoWriteBitstreamOp`

Operator class to write video bit stream to the disk.

This implementation is based on `nvidia::gxf::VideoWriteBitstream`.

##### Parameters

- **`output_video_path`**: The file path of the output video
  - type: `std::string`
- **`frame_width`**: The width of the output video
  - type: `int`
- **`frame_height`**: The height of the output video
  - type: `int`
- **`inbuf_storage_type`**: Input Buffer storage type, 0:kHost, 1:kDevice
  - type: `int`
- **`data_receiver`**: Receiver to get the data
  - type: `holoscan::IOSpec*`
