# AJA Source Operator

The AJA Source operator provides functionality to capture high-quality video streams from AJA capture cards and devices. It offers comprehensive support for both SDI (Serial Digital Interface) and HDMI (High-Definition Multimedia Interface) input sources, allowing for professional video capture in various formats and resolutions. The operator is designed to work seamlessly with AJA's hardware capabilities, including features like frame synchronization and format detection. Additionally, it provides an optional overlay channel capability that enables real-time mixing and compositing of multiple video streams, making it suitable for applications requiring picture-in-picture, graphics overlay, or other video mixing scenarios.

## Requirements

- AJA capture card (e.g., KONA HDMI)
- CUDA-capable GPU
- Holoscan SDK 1.0.3 or later

## Parameters

The following parameters can be configured for this operator:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `device` | string | Device specifier (e.g., "0" for device 0) | "0" |
| `channel` | NTV2Channel | Camera channel to use for input | NTV2_CHANNEL1 |
| `width` | uint32_t | Width of the video stream | 1920 |
| `height` | uint32_t | Height of the video stream | 1080 |
| `framerate` | uint32_t | Frame rate of the video stream | 60 |
| `interlaced` | bool | Whether the video is interlaced | false |
| `rdma` | bool | Enable RDMA for video input | false |
| `enable_overlay` | bool | Enable overlay channel | false |
| `overlay_channel` | NTV2Channel | Camera channel to use for overlay | NTV2_CHANNEL2 |
| `overlay_rdma` | bool | Enable RDMA for overlay | false |

## Supported Video Formats

The operator supports various video formats based on resolution, frame rate, and scan type:

- 720p (1280x720) at 50/59.94/60 fps
- 1080i (1920x1080) at 50/59.94/60 fps
- 1080p (1920x1080) at 23.98/24/25/29.97/30/50/59.94/60 fps
- UHD (3840x2160) at 23.98/24/25/29.97/30/50/59.94/60 fps
- 4K (4096x2160) at 23.98/24/25/29.97/30/50/59.94/60 fps

## Input Ports

- **overlay_buffer_input** (optional): Video buffer for overlay mixing when `enable_overlay` is true

## Output Ports

- **video_buffer_output**: Video buffer containing the captured frame
- **overlay_buffer_output** (optional): Empty video buffer for overlay when `enable_overlay` is true

## Enabling overlay

AJA source operator supports overlay workflows through its overlay input/output buffers.
Holoviz is a good candidate for this workflow because it can consume and return the render buffer directly.
In particular, it can render into an existing RGBA render buffer provided at `render_buffer_input` and, when `render_buffer_output` is enabled, return the filled render buffer while reusing the same input buffer instead of allocating a new one.
This makes it suitable for low-latency overlay paths and avoids an extra buffer allocation/copy in the common AJA overlay workflow, but it is not strictly required as long as another operator correctly handles the buffers provided by `aja_source`.

To enable overlay on the AJA source operator, set `enable_overlay: true`.

For Holoviz-based overlay workflows, configure the **Holoviz** operator with
`enable_render_buffer_input: true` and `enable_render_buffer_output: true`.
These are Holoviz settings (not `aja_source` parameters).

Two common usage patterns are:

1. **Overlay-only path (headless Holoviz)**

   Holoviz is configured with `headless: true` and is only used to render overlay content into the render buffer that is sent back to the AJA source operator.

   Required connections:

   ```python
   self.add_flow(source, visualizer, {("overlay_buffer_output", "render_buffer_input")})
   self.add_flow(visualizer, source, {("render_buffer_output", "overlay_buffer_input")})
   ```

2. **Holoviz displays the source image and the overlay**

   In this case, Holoviz is also used as the display sink for the captured source image, so the source image must be connected to Holoviz through `receivers`:

   ```python
   self.add_flow(source, visualizer, {("video_buffer_output", "receivers")})
   ```

   The overlay feedback loop is still required:

   ```python
   self.add_flow(source, visualizer, {("overlay_buffer_output", "render_buffer_input")})
   self.add_flow(visualizer, source, {("render_buffer_output", "overlay_buffer_input")})
   ```

In summary:

- `video_buffer_output` carries the captured source image to Holoviz for display (used in pattern 2 only).
- `overlay_buffer_output` provides the render buffer consumed by Holoviz.
- `render_buffer_output` returns the rendered overlay buffer back from Holoviz to the AJA source operator.

Any overlay tensors, masks, text, or geometry should still be connected to Holoviz according to the needs of the application graph.
