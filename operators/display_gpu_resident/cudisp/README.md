# cuDisp

cuDisp is a display library that integrates with the Linux DRM stack and CUDA. It provides swapchain creation, buffer acquisition, and present (page-flip) from host or GPU, with optional GPU-driven present via a notify path.

**IGX version:** IGX SW 2.1

**Limitations (v1.0):**

*Display mode / output:*

- `modeWidth` and `modeHeight` in `MODE_INFO` are currently ignored; mode selection is based solely on buffer dimensions provided in `BUFFER_INFO`. Buffer width and height must exactly match a supported mode on the connected display. Scaling and windowed mode are not supported.
- `maxBpc` values other than `CUDISP_MAX_BPC_DEFAULT` are not supported.
- `VSYNC_OFF` present flag is not supported.

*Layers / overlay planes:*

- Only layer 0 (primary plane) is supported. Overlay planes (`layerIndex > 0`) are not supported.
- Multi-layer present (`numLayers > 1`) is not supported.

*Surface formats:*

- Only `ARGB8888`, `XRGB8888`, and `ABGR16161616` are supported. All other formats return not supported.

*Scaling / positioning:*

- Scaling (`scaleWidth` / `scaleHeight` != 0) is not supported.
- Windowed mode (non-zero `posX` / `posY`) is not supported.

*Per-plane properties:*

- Non-opaque plane alpha (`alpha` != 0xFFFF) is not supported.
- Non-default blend mode is not supported.
- Rotation / reflection is not supported.
- Non-default color encoding is not supported.
- Non-default color range is not supported.

*Color management / HDR:*

- HDR metadata is not supported.
- Colorspace configuration is not supported.
- Degamma LUT, Gamma LUT, and CTM are not supported.

*Other:*

- Display selection by name is not supported; the first connected display is always used.
