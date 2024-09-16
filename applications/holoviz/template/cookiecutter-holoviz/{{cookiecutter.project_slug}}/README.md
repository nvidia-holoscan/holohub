# {{ cookiecutter.project_name }}

![]({{ cookiecutter.project_slug }}.png)

{%- if cookiecutter.example == "sRGB" %}
This application demonstrates the handling of the sRGB color space supported by the Holoviz operator.

The Holoviz operator can convert sRGB input images to linear color space before rendering and also can convert from linear color space to sRGB before writing to the frame buffer.

sRGB color space can be enabled for input images and for the frame buffer independently. By default, the sRGB color space is disabled for both.

By default, the Holoviz operator is auto detecting the input image format. Auto detection always assumes linear color space for input images. To change this to sRGB color space explicitly set the `image_format_` member of the input spec for that input image to a format ending with `SRGB`:

```cpp
    // By default the image format is auto detected. Auto detection assumes linear color space,
    // but we provide an sRGB encoded image. Create an input spec and change the image format to
    // sRGB.
    ops::HolovizOp::InputSpec input_spec("image", ops::HolovizOp::InputType::COLOR);
    input_spec.image_format_ = ops::HolovizOp::ImageFormat::R8G8B8_SRGB;

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}));
```

By default, the frame buffer is using linear color space. To use the sRGB color space, set the `framebuffer_srbg` argument of the Holoviz operator to `true`:

```cpp
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // enable the sRGB frame buffer
        Arg("framebuffer_srbg", true));
```

{%- elif cookiecutter.example == "vsync" %}
This application demonstrates the capability of the Holoviz operator to wait for the vertical blank of the display before updating the current image. It prints the displayed frames per second to the console, if sync to vertical blank is enabled the frames per second are capped to the display refresh rate.

To enable syncing to vertical blank set the `vsync` parameter of the Holoviz operator to `true`:

```cpp
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // enable synchronization to vertical blank
        Arg("vsync", true));
```

By default, the Holoviz operator is not syncing to the vertical blank of the display.
{%- elif cookiecutter.example == "YUV" %}
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

{%- endif %}

## Run Instructions

To build and start the application:

```bash
./dev_container build_and_run {{ cookiecutter.project_slug }}
```
