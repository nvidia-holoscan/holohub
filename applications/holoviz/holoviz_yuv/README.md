# Holoviz YUV

![](holoviz_yuv.png)
This application demonstrates the capability of the Holoviz operator to display images in YUV (aka YCbCr) format.

Holoviz supports multiple YUV formats including 420 and 422, 8 and 16 bit, single plane and multi plane. It supports BT.601, BT.709 and BT.2020 color conversions, narrow and full range and cosited even and midpoint chroma downsample positions.

The application creates a GXF video buffer containing YUV 420 BT.601 extended range data.

The YUV image properties are specified using a input spec structure:

```cpp
    ops::HolovizOp::InputSpec input_spec("image", ops::HolovizOp::InputType::COLOR);

    // Set the YUV image format, model conversion and range for the input tensor.
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM;
    input_spec.yuv_model_conversion_ = ops::HolovizOp::YuvModelConversion::YUV_601;
    input_spec.yuv_range_ = ops::HolovizOp::YuvRange::ITU_FULL;

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}));
```

## Run Instructions

To build and start the application:

```bash
./dev_container build_and_run holoviz_yuv
```
