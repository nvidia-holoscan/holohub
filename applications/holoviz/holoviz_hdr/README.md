# Holoviz HDR

![](holoviz_hdr.png)
This application demonstrates displaying HDR images using the Holoviz operator. The application creates image data in HDR10 (BT2020 color space) with SMPTE ST2084 Perceptual Quantizer (PQ) EOTF and displays the image on the screen.

Note that the screenshot above does not show the real HDR image on the display since it's not possible to take screenshots of HDR images.

The Holoviz operator parameter `display_color_space` is used to set the color space. This allows HDR output on Linux distributions and displays supporting that feature. See https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html#hdr for more information.

```cpp
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // select the HDR10 ST2084 display color space
        Arg("display_color_space", ops::HolovizOp::ColorSpace::HDR10_ST2084));
```

## Run Instructions

To build and start the application:

```bash
./holohub run holoviz_hdr
```
