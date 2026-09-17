# Holoscan Camera Release Notes

## Release Artifacts

For module version 0.2.0:

- 📕 **Documentation:** [Holoscan Camera setup and usage](README.md)

See the
[Holoscan SDK 5.0 EA2 Release Notes](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0-ea2/RELEASE_NOTES.md).

## 0.2.0

V4L2 capture qualification identified non-zero frame-quality counters on some
platforms. See the [V4L2 benchmarks and qualification results](docs/v4l2_capture_op_benchmarks.md)
for measurements, capture settings, and diagnostic limitations.

### Known Issues

| Issue | Description |
| --- | --- |
| 6744108 | `holoscan_camera_v4l2` reported degraded frames on IGX Thor Blackwell and one driver-flagged corrupt frame per recorded run on DGX Spark and x86 + Blackwell. The cause remains under investigation. Consumers should inspect sample quality flags and reject invalid frames; see [counter semantics](operators/v4l2_capture_op/README.md#capture-counters). |

## 0.1.0

Introduced the initial Holoscan SDK 5.x C++ interface stub for `V4l2CaptureOp`.
