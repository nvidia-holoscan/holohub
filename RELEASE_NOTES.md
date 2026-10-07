# Holoscan Camera Release Notes

## Release Artifacts

For module version 0.5.0:

- 📕 **Documentation:** [Holoscan Camera setup and usage](README.md)

See the
[Holoscan SDK 5.0 EA3 Release Notes](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0-ea3/RELEASE_NOTES.md).

## 0.5.0

This release adds `ArgusCaptureOp` for Jetson cameras and an installable module
surface for use inside the Holoscan SDK build.

### Holoscan CLI Setup

Before running `./holoscan_camera`, export both variables in the same shell
to select the EA3 CLI prerelease and its package indexes:

```bash
export HOLOSCAN_CLI_PINNED_VERSION="5.0.0a2"
export HOLOSCAN_CLI_INSTALL_ARGS="--pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/"
```

Use Python 3.11 or newer for Holoscan CLI 5. Run
`./holoscan_camera version` and confirm `5.0.0a2` before building or running
camera applications.

### New Features

#### Operators

- Added `ArgusCaptureOp` for Jetson Argus cameras, publishing
  ISP-processed frames and sensor metadata.

### Build and Packaging

- All install rules carry the `holoscan-camera` component.
- Nested, non-top-level builds install `holoscan-camera-config.cmake`,
  `holoscan-camera-config-version.cmake`, and `holoscan-camera-targets.cmake`
  under `<libdir>/cmake/holoscan`, using GNUInstallDirs.
- Nested builds no longer run CPack by default; set
  `HOLOSCAN_CAMERA_BUILD_DEB=ON` to produce the Debian package from a nested
  build.
- `holoscan-cameraConfig.cmake` finds the SDK itself when needed by a consumer
  that has not already configured it.
- Standalone Debian packaging remains unchanged.

### Bug Fixes

| Issue | Description |
| --- | --- |
| 6800245 | On L4T R38 systems, V4L2 container builds skip SIPL and Argus dependency downloads. CMake disables SIPL targets, while Argus retains its usual build options; the separate Argus configuration failures are listed under Known Issues. |
| — | Fixed SIPL buffer lifetimes and matched-metadata preservation. |
| — | Fixed SIPL RAW image offset and active height. |

### Known Issues

| Issue | Description |
| --- | --- |
| 6843559 | `holoscan_camera_v4l2` test configuration on IGX Thor iGPU / L4T R38.5 can fail on missing `Argus/Argus.h` because standalone builds enable unrelated Argus targets. The reported `test` command does not accept the `--configure-args` workaround used with `run`. |
| 6840016 | V4L2 configuration on DGX Spark can fail on the unavailable `nvargus_socketclient` library because unrelated Argus targets are enabled. |
| 6801049 | SIPL capture startup can fail on Jetson when NvSIPL cannot read `/proc/device-tree/model` or parse the camera database. |
| 6714027 | Stereo-synchronized SIPL capture can intermittently fail to start or stall. Validate stream health beyond the first few seconds; see the [0.3.0 finding](#030). |
| 6744108 | `holoscan_camera_v4l2` can report degraded or driver-flagged corrupt frames on some platforms. Inspect sample quality flags and reject invalid frames; see the [0.3.0 finding](#030) and [counter semantics](operators/v4l2_capture_op/README.md#capture-counters). |

## 0.4.0

This release moves SIPL capture to the post-EA2 Holoscan SDK APIs. Applications
using the 0.3.0 SIPL interface must update graph construction and rebuild
against the SDK revision documented below.

### Breaking changes

`SIPLCaptureOp` now requires its shared `SIPLCaptureService` and camera index
at construction. `SIPLCaptureOpParams` and the corresponding `set_params()`
call have been removed. Update graph construction from:

```cpp
auto camera = graph.op<SIPLCaptureOp>("camera", holoscan::MemoryKind::kHost);
camera.set_params(SIPLCaptureOpParams{.service = service, .camera_index = 0U});
```

to:

```cpp
auto camera = graph.op<SIPLCaptureOp>(
    "camera", service, /*camera_index=*/0U, holoscan::MemoryKind::kHost);
```

The service must be non-null. `memory_kind` remains an optional final argument
and defaults to `holoscan::MemoryKind::kCudaDevice`. Multiple operators can
share the same service; graph constructor captures retain its shared lifetime.

### SDK compatibility

Use Holoscan SDK 5.0 RC5, which supports native shared-pointer constructor
captures. SDK 5.0 EA2 rejects that argument to `Graph::op` at compile time.
The SDK package version is still `5.0.0` across these source revisions, so
checking that version alone cannot
enforce this requirement.

SIPL schema generation also uses the newer `SCHEMA`, `COMPATIBILITY_EPOCH`
and `MAX_SERIALIZED_SIZE` arguments, which the EA2 SDK's schema helper does
not accept. Both schema generation and constructor captures require the newer
SDK; changing only the application constructor call is insufficient.

### Build fixes

Fixed duplicate CMake argument-keyword warnings when configuring the Debian
package from a parent scope that already defines `oneValueArgs`.

### Known issues

The SIPL stereo startup/stall finding 6714027 and V4L2 frame-quality finding
6744108 documented under [0.3.0](#030) remain applicable.

## 0.3.0

This release targets Holoscan SDK 5.0 EA2 and retains the
`SIPLCaptureOpParams` / `set_params()` interface.

Added `SIPLCaptureOp`: event-driven RAW10/NV12 capture for NvSIPL-managed
cameras on Jetson/IGX platforms, ported from Holoscan Sensor Bridge, with
per-camera NvSci sync handling for multi-camera rigs. Adds two applications
built on the operator (`sipl_frame_saver`, `sipl_stereo_monitor`) and the
container/build support needed to compile against NvSIPL. See the
[operator guide](operators/sipl_capture_op/README.md) for supported
JetPack/L4T versions, ports, and usage.

### Known Issues

| Issue | Description |
| --- | --- |
| 6714027 | Stereo-synchronized SIPL capture (`sync_sensors: true`, HSB vb1940 UDDF driver) shows an intermittent startup/stall failure rate (observed 3-19% across recorded runs). Reproduced with `nvsipl_camera` alone, bypassing HSDK5/HSB/holoscan-camera entirely, so the root cause is upstream of `SIPLCaptureOp`. See the [SIPL benchmarks and qualification results](docs/sipl_capture_op_benchmarks.md) for recorded rates and camera-firmware notes. Consumers relying on stereo sync should validate stream health past the first few seconds rather than assume a clean start implies a clean run. |
| 6744108 | `holoscan_camera_v4l2` reported degraded frames on IGX Thor Blackwell and one driver-flagged corrupt frame per recorded run on DGX Spark and x86 + Blackwell. The cause remains under investigation. Consumers should inspect sample quality flags and reject invalid frames; see [counter semantics](operators/v4l2_capture_op/README.md#capture-counters). |

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
