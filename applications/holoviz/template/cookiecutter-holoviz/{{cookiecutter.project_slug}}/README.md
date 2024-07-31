# {{ cookiecutter.project_name }}

![]({{ cookiecutter.project_slug }}.png)<br>

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

{%- endif %}
{%- if cookiecutter.example == "vsync" %}
This application demonstrates the capability of the Holoviz operator to wait for the vertical blank of the display before updating the current image. It prints the displayed frames per second to the console, if sync to vertical blank is enabled the frames per second are capped to the display refresh rate.

To enable syncing to vertical blank set the `vsync` parameter of the Holoviz operator to `true`:

```cpp
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // enable synchronization to vertical blank
        Arg("vsync", true));
```

By default, the Holoviz operator is not syncing to the vertical blank of the display.
{%- endif %}

## Run Instructions

To build and start the application:

```bash
./dev_container build_and_run {{ cookiecutter.project_slug }}
```
