# Depth to Point Cloud Demo

A minimal, hardware-free demo of the [`depth_to_point_cloud`](../../operators/depth_to_point_cloud)
operator. It generates a synthetic organized depth image (a gently tilting plane) plus an aligned
RGB image entirely on the GPU, deprojects it into an organized `H x W x 3` point cloud, and reports
the valid-point count and Z range each frame. No camera or dataset is required.

## Run

```bash
./holohub run depth_to_point_cloud_demo
# or directly:
python3 applications/depth_to_point_cloud_demo/depth_to_point_cloud_demo.py --frames 100
```

Expected output (per frame):

```text
[depth_to_point_cloud_demo] points=307200 valid=307200 z=[1.xxx, 2.xxx] m
```

## Pipeline

```text
SyntheticDepthGeneratorOp  --depth-->  DepthToPointCloudOp  --point_cloud-->  PointCloudStatsOp
                           --color-->
```

## Using a real Intel RealSense camera

Replace the synthetic generator with the [`realsense_camera`](../../operators/realsense_camera)
operator. It applies librealsense's `units_transform` internally, so `depth_buffer` is emitted as
**`GRAY32F` float32 already in meters** (in a `VideoBuffer`), and `color_buffer` as `RGBA8`. Because
the depth is metric, use `depth_scale=1.0` — **not** `0.001`; the `0.001` (uint16 millimeters) value
is only for raw `Z16` sources that have not been unit-transformed:

```python
from holohub.realsense_camera import RealsenseCameraOp

camera = RealsenseCameraOp(self, name="camera")
cloud = DepthToPointCloudOp(
    self, name="point_cloud", allocator=...,
    fx=fx, fy=fy, cx=cx, cy=cy,   # from the camera's depth_camera_model intrinsics
    depth_scale=1.0,              # depth_buffer is float32 meters (units_transform applied)
    depth_min=0.1, depth_max=10.0,
)
self.add_flow(camera, cloud, {("depth_buffer", "depth")})
self.add_flow(camera, cloud, {("color_buffer", "color")})  # color_channels = 4 (RGBA)
```

The camera also exposes `depth_camera_model` / `color_camera_model` outputs carrying the per-stream
intrinsics, which can drive the optional `intrinsics` input instead of static `fx/fy/cx/cy`.

## Interactive 3D visualization

To view the cloud, replace `PointCloudStatsOp` with `HolovizOp` configured to render the
`point_cloud` tensor as 3D points (and the `colors` tensor for per-point color). Reshaping the
organized `H x W x 3` cloud to `N x 3` and dropping invalid (NaN) points first is recommended.

## Requirements

- Holoscan SDK ≥ 4.0.0, CUDA, CuPy. Builds the `depth_to_point_cloud` operator (declared as a
  dependency). Platforms: `x86_64`, `aarch64`.
