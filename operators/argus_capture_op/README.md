# ArgusCaptureOp

Holoscan SDK 5 source for ISP-backed cameras on Jetson AGX Orin exposed through NVIDIA `libargus`.
Each operator owns one capture session. Multiple operators share the process's
Argus camera provider and CUDA primary context.

## Platform and dependencies

| Platform target | Capture requirements | Validation in this change |
| --- | --- | --- |
| Jetson AGX Orin | Compatible Jetson Linux BSP, CUDA and HSDK 5 build | Initial hardware capture validation passed; results below |

Hardware validation on Jetson AGX Orin captured 300 frames at 1920x1080 NV12,
30 FPS, with timestamps and exposure metadata, zero reported drops or degraded
frames, and a clean exit (`status=0`). Saving a captured frame and converting it
to PNG also succeeded.

A camera must appear in libargus's device/mode enumeration. MIPI CSI-2 and GMSL
cameras require the manufacturer's matching kernel driver, device-tree setup,
serializer/deserializer configuration where applicable, and ISP tuning. A V4L2
device alone does not establish Argus support. Transport setup belongs to the BSP;
the operator selects the camera exposed by Argus.

Install the **Jetson Multimedia API development package** matching the device's
BSP (`nvidia-l4t-jetson-multimedia-api`), the libargus socket-client library,
NvBufSurface, EGL development files, and CUDA. The `nvargus-daemon` service
must be running for capture. Builds require the headers and direct link
libraries and fail configuration if they are missing.

`libEGL.so` is the standard [EGL library](https://github.com/NVIDIA/libglvnd#egl-dispatching).
The native backend initializes an `EGLDisplay` and shares `NvBufSurface`
allocations as `EGLImage` buffers with Argus and CUDA. This dependency is required
for capture, including headless runs, independently of downstream HoloViz
visualization. Live capture requires the NVIDIA EGL runtime supplied by the
matching Jetson BSP.

Use a matched Jetson AGX Orin BSP, CUDA, and HSDK 5 toolchain that satisfies
HSDK's CUDA and OS requirements.

## Build

The operator and sample build only for ARM64 targets (`aarch64` or `arm64`).
`OP_argus_capture_op` enables the operator, and `APP_holoscan_camera_argus` enables
the sample and its operator dependency. `BUILD_ALL=ON` includes them on ARM64,
including R38. An R38 build requires matching development headers and link
libraries; it does not qualify Argus capture on IGX Thor. For V4L2-only R38
builds, use `-DBUILD_ALL=OFF -DAPP_holoscan_camera_v4l2=ON`. Other
architectures skip the targets, and explicitly enabling either there produces
a configuration error.

On ARM64 R39 and later, the project Dockerfile installs Argus/NvBufSurface
headers from NVIDIA's Multimedia API package using the same `L4T_VERSION`
build argument as SIPL. It
also extracts `libnvargus_socketclient` and `libnvbufsurface` for linking, without
installing the BSP. Argus dependency installation is skipped on R38 and other
architectures. For experimental IGX R38.5 work, mount a complete host
Multimedia API development tree explicitly when it is available:

```bash
./holoscan_camera run-container --docker-opts \
  '--mount type=bind,source=/usr/src/jetson_multimedia_api,target=/usr/src/jetson_multimedia_api,readonly'
```

`--docker-opts` replaces the mode's default Docker options, so include any
other options required by the selected mode. The header mount does not supply
the Argus and NvBufSurface link libraries needed for an Argus build.
The resolved downloaded package versions and checksums are recorded in
`/usr/share/holoscan-camera/argus-build-packages.txt` in the image.
Native Argus builds and their consumers use `--allow-shlib-undefined` for the
libraries' transitive BSP dependencies, which are supplied by the runtime device.
The direct dependencies remain recorded in the resulting shared library.

To build all operators and applications together on R39 or later, run inside
the project build container for ARM64 with the HSDK install tree available:

```bash
cmake -S . -B build-all -G Ninja \
  -Dholoscan_ROOT=/path/to/holoscan/install \
  -DBUILD_ALL=ON -DHOLOSCAN_CAMERA_BUILD_TESTING=ON
cmake --build build-all -j4
```

For an Argus-only build:

```bash
cmake -S . -B build-argus -G Ninja \
  -Dholoscan_ROOT=/path/to/holoscan/install \
  -DBUILD_ALL=OFF -DAPP_holoscan_camera_argus=ON \
  -DHOLOSCAN_CAMERA_BUILD_TESTING=ON
cmake --build build-argus -j4
ctest --test-dir build-argus --output-on-failure
```

The operator requires current HSDK 5 schema tooling with inferred traits and
`COMPATIBILITY_EPOCH` support, as used by this repository's SIPL operator.

Custom development installations can set `ARGUS_INCLUDE_DIR`,
`NVBUFSURFACE_INCLUDE_DIR`, `ARGUS_EGL_INCLUDE_DIR`, `ARGUS_LIBRARY`,
`NVBUFSURFACE_LIBRARY`, and `ARGUS_EGL_LIBRARY`.

For container capture on Jetson AGX Orin, build with `L4T_VERSION` matching the
host BSP and supply the host's Jetson libraries/devices and Argus daemon socket
(`/tmp/argus_socket`) using the NVIDIA container runtime. Start `nvargus-daemon`
on the host before running the [sample](../../applications/holoscan_camera_argus/README.md).
The extracted build libraries alone do not provide a camera runtime. EGL device
displays are tried first, so an X11 display is not required. The sample's
`--validate --memory host` mode checks graph compilation on the Jetson runtime
without opening a camera.

## Graph interface

```cpp
#include <argus_capture_op/argus_capture_op.hpp>

using namespace holoscan::holoscan_camera;
ArgusCaptureConfig config;
config.camera_index = 0;
config.sensor_mode = 0;
config.width = 1920;
config.height = 1080;
config.fps = 30;
auto camera = graph.op<ArgusCaptureOp>("camera", config);
graph.add_flow(camera->frame, consumer->image);

// Apply these options when calling holoscan::compile(graph, options).
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "camera", .output_port = "frame", .device = holoscan::DeviceId{0}});
```

Link `holoscan::argus_capture_op`. Configuration is a constructor value, so graph
authoring and runtime graph compilation open no camera. Native negotiation and
allocation happen during the operator lifecycle.

| Port | Payload | Contents |
| --- | --- | --- |
| `frame` | `holoscan::schema::ImageT` | Rank-1 byte tensor, image dimensions, encoding, explicit plane offsets/strides, coordinate frame, device sequence, exposure and combined gain |
| `metadata` | `ArgusFrameMetadataT` | Camera/mode/source identity, sequence, source timestamps, exposure, frame duration, analog and ISP gains, AE/AWB status, loss and timeout counters |

Both ports use the same sequence in `EmitOptions::frame_id`. Calibrated capture
times, when enabled, also match. Publication across two ports is **not atomic**:
a full consumer queue may lose a frame or its metadata independently. Match by
sequence and tolerate missing partners; do not pair samples merely by arrival
order. The image descriptor carries essential image/settings information itself.

## Configuration

| Field | Default | Behavior |
| --- | --- | --- |
| `camera_index`, `sensor_mode` | 0, 0 | Indices reported by Argus; not `/dev/videoN` numbering |
| `width`, `height` | 1920, 1080 | Even dimensions from 2 to 16384; must not exceed the selected mode |
| `fps` | 30 | Fractional rates accepted; requested duration must lie within the sensor mode's range |
| `pixel_format` | `kNv12` | `kNv12` or `kI420`, subject to the platform's native buffer support |
| `buffer_count` | 6 | 3–64 native buffers, allocated once per resource epoch |
| `drop_policy` | `kDropOldest` | `kDropOldest` or `kDropNewest` for the operator's completed-frame queue |
| `timeout_ms` | 2000 | 100–60000 ms; must cover one frame period; maximum continuous interval without a frame |
| `cuda_device` | 0 | Jetson integrated GPU; other ordinals rejected |
| `memory_kind` | `kCudaDevice` | Optional `kHost` or `kPinnedHost` makes the transfer to host explicit |
| `frame_id` | `camera_optical_frame` | Coordinate frame name, 1–256 bytes |
| `integration_start_offset_ns` | unset | Optional sensor-calibrated signed correction from VI SOF to integration start |

NV12 packs Y followed by interleaved U/V; I420 packs Y, U, then V. Both produce
`width * height * 3 / 2` bytes. Native plane counts, dimensions, element types,
and color order are checked before capture. Unsupported format/mode combinations
fail rather than silently changing format. Raw Bayer, P010, RGB conversion,
runtime control updates, and synchronized multi-camera sessions are not included.
Color-space metadata remains unspecified because this implementation does not
establish the ISP's colorimetry.

## Memory and backpressure

```text
camera / ISP -> fixed NvBufSurface pool -> CUDA-mapped EGL image
            -> one device-to-device copy -> HSDK tensor pool -> downstream
```

The native backend uses `STREAM_TYPE_BUFFER`, `BUFFER_TYPE_EGL_IMAGE`, enabled
metadata, and `SYNC_TYPE_NONE`. It registers CUDA mappings once per native
buffer. It never maps image pixels to the CPU on the GPU path. Plane copies
remove native padding and complete on the HSDK write guard's producer stream before the buffer
returns to Argus. This is **one full-image GPU copy, not zero-copy publication**.
The HSDK allocation owns the published pixels; consumers cannot hold up reuse
of a native buffer. Explicit host placements use device-to-host copies.

The completed-frame queue holds at most `buffer_count - 2` frames. One slot is
reserved for compute and another for acquisition progress. A full queue drops
its oldest entry or rejects the newest acquisition according to `drop_policy`;
the rejected native buffer is returned immediately. Capture memory cannot grow
with consumer delay. HSDK tensor pools and graph edge queues are separately
bounded by the compiled plan. Pool exhaustion and publication backpressure drop
the current frame; partial fan-out publication is never retried. Counters in
metadata distinguish queue drops from publication drops. Sequence gaps mark
subsequent observed samples degraded.

## Timestamp meaning

The generic Argus sensor timestamp marks **first data arrival from the sensor**.
HSDK's `capture_time` means **start of integration**. These are not generally the
same instant. The operator therefore preserves the original timestamp without
silently substituting it for HSDK integration time.

- `sensor_timestamp_ns`: unchanged `ICaptureMetadata::getSensorTimestamp()`;
  its clock is BSP-defined and this operator does not assume it is POSIX monotonic.
- `sensor_sof_timestamp_ns`: `ISensorTimestampTsc::getSensorSofTimestampTsc()`
  when available, in **Tegra TSC nanoseconds**, not counter ticks. The explicit
  `timestamp_clock` enum identifies this domain; unavailable extension data is
  zero/UNKNOWN. A frame with neither source timestamp is a capture error.
- By default, `ImageT.header.capture_timestamp_ns` is zero and the HSDK envelope's
  `capture_time` is absent. The `metadata` output carries the source timestamps.
- If sensor timing has been calibrated, set `integration_start_offset_ns` so
  `integration_start = VI_SOF + offset`. The operator uses Sensor I/O's
  `ClockDiscipline` to project that corrected timestamp into the graph clock.
  The image, metadata, and sample envelope then agree. Missing TSC support or a
  failed clock projection is reported as an error in this mode.

Do not enable a zero offset merely to fill the timestamp field. Calibration must
remain valid for the configured exposure/readout behavior; automatic exposure
can invalidate a fixed correction. Projection uses the ARM generic timer counter
and frequency, with a one-time offset measurement per start. It requires an
unscaled realtime graph clock and does not compensate long-term clock drift.

References: NVIDIA's [capture metadata interface](https://docs.nvidia.com/jetson/archives/r39.2/ApiReference/classArgus_1_1ICaptureMetadata.html),
[TSC timestamp extension](https://docs.nvidia.com/jetson/archives/r38.2/ApiReference/classArgus_1_1Ext_1_1ISensorTimestampTsc.html),
and [buffer ownership API](https://docs.nvidia.com/jetson/l4t-multimedia/classArgus_1_1IBuffer.html).

## Lifecycle and failures

`kConfigure` resets clock state; `kAllocate` discovers the camera, validates the
mode, creates the session and native pool; `kArm` acquires notification authority;
`kStart` starts capture and its worker. `kStop` joins the worker, returns queued
buffers, cancels requests, waits for idle, and drains stale completions.
`kRelease` destroys mappings, buffers, session, and shared runtime references.
Sensor I/O `StageGuards` protects teardown after partial startup.

The worker requests a maximum 100 ms wait per acquire call
(`kAcquirePollTimeoutMs`). This polling interval balances stop-request and
notification-retry responsiveness against roughly ten timeout wakeups per second
on an idle stream. It is an operator scheduling choice, independent of the sensor
frame rate; acquire returns early when a frame is available. Transient acquire
timeouts are retried and counted. The separate `timeout_ms` setting controls the
continuous no-frame failure deadline and must cover at least one poll and one
frame period. Failure-notification retries use a 1 ms backoff
(`kNotificationRetryDelay`) to avoid busy-spinning during the start barrier or
backpressure while retrying promptly.

No frame for the configured continuous timeout fails the graph with `kTimeout`. Native errors,
missing metadata, invalid configuration, and copy failures produce contextual
messages plus failed lifecycle status or an HSDK compute error. Lost cameras
require application-directed recovery; this operator does not silently restart
the daemon or reopen a different sensor. Vendor calls may still block inside a
broken BSP; the public APIs do not provide forced driver cancellation.

On the inspected HSDK build, notification generations change on restart while
sender acquisition is only admitted at `kArm`. Request restarts at `kArm` or
earlier; a `kStart`-only restart can leave a stale sender and is diagnosed. A
fresh full run always creates a fresh session. Multi-camera capture is supported
through separate operators but does not imply synchronized sensor exposure.

## Validation

The unit suite uses the production operator with a deterministic test backend.
It covers typed image and metadata publication, NV12/I420 layouts, calibrated
timestamp correlation, bounded buffering under a slow consumer, allocation/start
failures, stream loss, copy failure, and persistent timeouts.
These checks do not replace hardware capture qualification.

On hosts without the Jetson Argus runtime, the
container CTest driver sets `HOLOSCAN_CAMERA_RUN_ARGUS_TESTS=OFF`, retaining
compilation while skipping Argus test registration and discovery. Direct CMake
builds default this option to `ON`. Run the suite on a compatible Jetson:

```bash
ctest --test-dir build-argus -L argus --output-on-failure
```

Manual device capture validation on a compatible Jetson remains required;
compilation alone does not qualify capture behavior.

Before claiming a platform/camera combination as supported, run the
[sample](../../applications/holoscan_camera_argus/README.md) on that combination:
verify supported modes/formats and actual cadence, sustained capture, slowdown
and drops, Ctrl-C cleanup and immediate recapture, daemon/device loss, and
sensor-specific timing against an external reference. Repeat for every platform
and BSP listed in the intended support matrix.
