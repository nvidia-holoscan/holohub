# cuDisp

> [!IMPORTANT]
> **cuDisp is an experimental proof of concept built for demo purposes.** It is not an official NVIDIA release and is published here only to accompany a sample application.
>
> - **No support commitments.** Issues and pull requests are reviewed on a best-effort basis only.
> - **No backward-compatibility guarantees.** The API, ABI, build, and behavior may change or be removed at any time without notice.
> - **Not a standalone library.** cuDisp is not packaged, distributed, or recommended for use outside the accompanying sample.

The cuDisp API in this sample displays CUDA buffers on screen via the Linux DRM stack. It exposes swapchain creation, buffer acquisition, and present (page-flip) from host or GPU, with optional GPU-driven present via a notify path.

**IGX version:** IGX SW 1.1.2

**Limitations:**

*G-SYNC (VRR) support:*

- G-SYNC is only supported for NVIDIA driver version 590.48.01 or higher.

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
