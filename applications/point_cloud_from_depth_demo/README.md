# Point Cloud From Depth Demo

A minimal, hardware-free demo of the [`point_cloud_from_depth`](../../operators/point_cloud_from_depth)
operator. It generates a synthetic organized depth image (a gently tilting plane) plus an aligned
RGB image entirely on the GPU, deprojects it into an organized `H x W x 3` point cloud, and reports
the valid-point count and Z range each frame. No camera or dataset is required.

## Run

```bash
./holohub run point_cloud_from_depth_demo
# or directly:
python3 applications/point_cloud_from_depth_demo/point_cloud_from_depth_demo.py --frames 100
```

Expected output (per frame):

```
[point_cloud_from_depth_demo] points=307200 valid=307200 z=[1.xxx, 2.xxx] m
```

## Pipeline

```
SyntheticDepthGeneratorOp  --depth-->  PointCloudFromDepthOp  --point_cloud-->  PointCloudStatsOp
                           --color-->
```

## Using a real Intel RealSense camera

Replace the synthetic generator with the [`realsense_camera`](../../operators/realsense_camera)
operator, which emits a `depth_buffer` (Z16 / `uint16` millimeters) and a `color_buffer` (RGBA8):

```python
from holohub.realsense_camera import RealsenseCameraOp

camera = RealsenseCameraOp(self, name="camera")
cloud = PointCloudFromDepthOp(
    self, name="point_cloud", allocator=...,
    fx=fx, fy=fy, cx=cx, cy=cy,   # from the camera's depth intrinsics
    depth_scale=0.001,            # RealSense depth is uint16 millimeters
    depth_min=0.1, depth_max=10.0,
)
self.add_flow(camera, cloud, {("depth_buffer", "depth")})
self.add_flow(camera, cloud, {("color_buffer", "color")})  # color_channels = 4 (RGBA)
```

## Interactive 3D visualization

To view the cloud, replace `PointCloudStatsOp` with `HolovizOp` configured to render the
`point_cloud` tensor as 3D points (and the `colors` tensor for per-point color). Reshaping the
organized `H x W x 3` cloud to `N x 3` and dropping invalid (NaN) points first is recommended.

## Requirements

- Holoscan SDK ≥ 3.5.0, CUDA, CuPy. Builds the `point_cloud_from_depth` operator (declared as a
  dependency). Platforms: `x86_64`, `aarch64`.
