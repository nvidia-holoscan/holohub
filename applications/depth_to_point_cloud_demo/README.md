# Depth to Point Cloud Demo

A minimal demo of the [`depth_to_point_cloud`](../../operators/depth_to_point_cloud) operator.

Here **hardware-free** means the input is produced by an on-GPU synthetic data generator
(`SyntheticDepthGeneratorOp`) rather than a physical sensor: no depth camera, recorded dataset, or
depth-estimation network is required, so the demo runs in CI on any GPU. The generator emits a
synthetic organized depth image (a gently tilting plane) plus an aligned RGB image entirely on the
GPU; the operator deprojects it into an organized `H x W x 3` point cloud, and the demo reports the
valid-point count and Z range each frame.

## Run

```bash
# Synthetic source (default, hardware-free, CI-friendly):
./holohub run depth_to_point_cloud_demo
# equivalently:
./holohub run depth_to_point_cloud_demo synthetic

# or directly:
python3 applications/depth_to_point_cloud_demo/depth_to_point_cloud_demo.py --frames 100
```

Expected output (per frame):

```text
[depth_to_point_cloud_demo] points=307200 valid=307200 z=[1.xxx, 2.xxx] m
```

### Sources (`--source` / run modes)

The demo selects its input source with `--source`, exposed as HoloHub run modes:

| Mode / `--source` | Description |
| --- | --- |
| `synthetic` (default) | On-GPU synthetic depth + RGB generator. No hardware, runs in CI. |
| `realsense` | Live Intel RealSense camera (see caveat below). |

### Interactive 3D visualization (`--visualize`)

By default the demo ends in a headless, CI-friendly `PointCloudStatsOp` sink that just reports
per-frame statistics. Pass `--visualize` to instead render the cloud in `HolovizOp` as 3D points:

```bash
python3 applications/depth_to_point_cloud_demo/depth_to_point_cloud_demo.py --visualize
```

With `--visualize`, the organized `H x W x 3` cloud is compacted to `N x 3` with invalid (NaN)
points dropped before being handed to `HolovizOp` as a `points_3d` primitive. This path needs a
display and is therefore disabled by default (the CI mode keeps the statistics sink).

## Pipeline

```text
SyntheticDepthGeneratorOp  --depth-->  DepthToPointCloudOp  --point_cloud-->  PointCloudStatsOp
                           --color-->                                         (or HolovizOp with --visualize)
```

## Using a real Intel RealSense camera

`--source realsense` is scaffolded but **not yet runnable from this Python demo**. The
[`realsense_camera`](../../operators/realsense_camera) operator is currently C++-only (it ships no
Python bindings), and it emits its `depth_buffer` / `color_buffer` as GXF `VideoBuffer`s, whereas
`DepthToPointCloudOp` consumes a GXF `Tensor`. Wiring them from Python therefore requires:

1. Python bindings for `realsense_camera` (add an `operators/realsense_camera/python/` module).
2. A `FormatConverterOp` between the camera and the operator to convert `VideoBuffer` → `Tensor`
   (and `RGBA8` → `RGB8` for the color path).
3. Feeding intrinsics — either statically (`fx/fy/cx/cy`) or by consuming the camera's
   `depth_camera_model` output through the operator's optional `intrinsics` input.

Selecting `--source realsense` today raises a clear `NotImplementedError` pointing here. The wiring,
once the bindings exist, is:

```python
from holohub.realsense_camera import RealsenseCameraOp  # requires new Python bindings

camera = RealsenseCameraOp(self, name="camera", allocator=...)
# camera.depth_buffer / color_buffer are VideoBuffers -> convert to Tensor via FormatConverterOp,
# then feed DepthToPointCloudOp. RealSense depth is float32 meters (units_transform applied), so
# use depth_scale=1.0 — not 0.001 (the 0.001 uint16-millimeter value is only for raw Z16 sources).
```

## Requirements

- Holoscan SDK ≥ 4.0.0, CUDA, CuPy. Builds the `depth_to_point_cloud` operator (declared as a
  dependency). Platforms: `x86_64`, `aarch64`.
