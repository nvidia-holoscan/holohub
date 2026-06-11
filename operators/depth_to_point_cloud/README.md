# Depth to Point Cloud Operator

Deproject an organized depth image into an organized point cloud on the GPU. This is the
"gateway to 3D" building block: it turns a depth image (from a depth camera, stereo matcher,
or monocular depth network) plus pinhole intrinsics into per-pixel `XYZ` points that downstream
operators (mapping, ground-plane / traversability estimation, registration, Holoviz 3D rendering,
PCL/Open3D interop) can consume — all GPU-resident, zero-copy on the hot path.

## What it computes

For each pixel `(u, v)` with metric depth `Z = raw_depth * depth_scale`, the output point in the
**camera optical frame** (x-right, y-down, z-forward) is:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

The kernel is a single map/elementwise pass (one CUDA thread per pixel, grid-stride, coalesced).
Pixels with `raw_depth == 0`, a non-finite value, or a metric depth outside `[depth_min, depth_max]`
are written as `(invalid_value, invalid_value, invalid_value)` (default `NaN`), keeping the output
**organized** (`H x W x 3`) so pixel neighborhoods are preserved for downstream normals/segmentation.

> Frame note: the output is in the optical frame. For a ROS body frame (x-forward, y-left, z-up)
> apply the standard optical→body rotation downstream.

## Ports

| Port | Direction | Type | Notes |
| --- | --- | --- | --- |
| `depth` | in | `Entity` w/ 2D tensor | `uint16` (scaled by `depth_scale`) or `float32` (meters at `depth_scale=1.0`), shape `[H, W]` or `[H, W, 1]`, device memory |
| `intrinsics` | in (optional) | `Entity` w/ `float32[4]` | `[fx, fy, cx, cy]`; overrides the params for that frame |
| `color` | in (optional) | `Entity` w/ `uint8` image | `[H, W, 3]` or `[H, W, 4]` aligned to depth; enables colored output |
| `point_cloud` | out | `Entity` | `float32 [H, W, 3]` XYZ (and `uint8 [H, W, 3]` RGB colors when `color` is connected) |

The emitted `colors` tensor is always 3-channel RGB; a 4-channel (RGBA) `color` input is converted to
RGB and its alpha channel is dropped.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `fx`, `fy`, `cx`, `cy` | `0.0` | Pinhole intrinsics in pixels (used when `intrinsics` port is unconnected) |
| `depth_scale` | `0.001` | Raw-depth → meters multiplier (`0.001` for uint16 mm; `1.0` for float32 m) |
| `depth_min`, `depth_max` | `0.0`, `100.0` | Valid metric depth range (meters) |
| `invalid_value` | `NaN` | Value written to X/Y/Z for invalid pixels |
| `depth_tensor_name` / `color_tensor_name` | `""` | Input tensor names (empty = first tensor) |
| `output_tensor_name` | `"point_cloud"` | Name of the emitted XYZ tensor |
| `output_color_tensor_name` | `"colors"` | Name of the emitted colors tensor |
| `allocator` | — | Device allocator for the output tensors (e.g. `BlockMemoryPool`) |

## Usage

### Python

```python
from holohub.depth_to_point_cloud import DepthToPointCloudOp

cloud = DepthToPointCloudOp(
    self,
    name="point_cloud",
    allocator=BlockMemoryPool(self, ...),
    fx=fx, fy=fy, cx=cx, cy=cy,
    depth_scale=0.001,          # uint16 millimeters
    depth_min=0.1, depth_max=10.0,
)
# depth_source -> cloud -> HolovizOp (3D points)
```

### C++

```cpp
auto cloud = make_operator<ops::DepthToPointCloudOp>(
    "point_cloud",
    Arg("allocator", make_resource<BlockMemoryPool>(...)),
    Arg("fx", fx), Arg("fy", fy), Arg("cx", cx), Arg("cy", cy),
    Arg("depth_scale", 0.001f));
```

## Testing

`test/test_deproject.cu` is a standalone golden-reference unit test that depends only on the CUDA
runtime (no Holoscan SDK). It verifies the deprojection math (float32 and uint16 paths), invalid /
out-of-range handling, and color passthrough against an analytic CPU reference:

```bash
nvcc -O2 -arch=native -o test_deproject test/test_deproject.cu deproject.cu && ./test_deproject
```

It is also registered with CTest (`depth_to_point_cloud_test`) when the project is built with
`BUILD_TESTING` enabled.

## Requirements

- Holoscan SDK ≥ 4.0.0, CUDA. Platforms: `x86_64`, `aarch64` (Jetson). No third-party dependencies
  beyond the CUDA runtime — the deprojection runs as a custom CUDA kernel (no VPI/PVA/VIC fixed-function
  block exists for depth deprojection, and it is not a library-shaped op).
