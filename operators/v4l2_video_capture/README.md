### V4L2 Video Capture

The `v4l2_video_capture` operator provides support for USB and HDMI.

#### `holoscan::ops::V4L2VideoCapture`

This implementation is based on `nvidia::gxf::V4L2VideoCapture`.

##### Parameters

- **`allocator`**: Output Allocator
  - type: `holoscan::Allocator*`
- **`device`**: Path to the V4L2 device
  - type: `string`
- **`width`**: Width of the V4L2 image
  - type: `int32`  
- **`height`**: Height of the V4L2 image
  - type: `int32`
- **`num_buffers_`**: Number of V4L2 buffers to use
  - type: `int32`    
- **`pixel_format_`**: Pixel format of capture stream (RGBA32 or YUYV)
  - type: `string`   