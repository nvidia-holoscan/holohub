# sipl_capture_op

Holoscan SDK 5.x camera capture operator for NvSIPL-managed cameras on
Jetson/IGX platforms. Ported from Holoscan Sensor Bridge's `SIPLCaptureService`,
wired to HSDK5 lifecycle hooks, `holoscan::sensor_io` schemas, and a configurable
frame placement (host or CUDA-device). Requires NVIDIA JetPack NvSIPL dev
packages (`NvSIPLCamera.hpp`) to build. No dependency on Holoscan Sensor Bridge,
with exceptions:

1. the SIPL driver stack may use HSB as one possible underlying driver, not
   directly exposed to `SIPLCaptureOp`;
2. SIPL frames from HSB inputs leak some HSB-specific metadata details. We
   handle these directly with static metadata details derived from the HSB
   project sources.

## Files

| File | Purpose |
| --- | --- |
| `sipl_capture_op.hpp` / `.cpp` | `SIPLCaptureOp` — one operator instance per camera |
| `sipl_capture_service.hpp` / `.cpp` | `SIPLCaptureService` — owns the NvSIPL pipeline; shared across cameras in a rig |
| `sipl_compat.hpp` | SIPL API v1/v2 compatibility shims (`NVSIPL_API_MAJOR_VERSION`) — see [Supported versions](#supported-versions) |
| `hsb_frame_metadata.hpp` | Parses HSB metadata embedded in the frame buffer |
| `tsc_to_ns.hpp` | Converts the Tegra TSC tick counter to nanoseconds |
| `sipl_frame_metadata.fbs` | FlatBuffers schema for the `sensor_data` output port |
| `tests/` | Unit tests (no camera needed) and hardware integration tests |

## Supported versions

Only the latest JetPack BSP (7.2.1 / L4T R39, SIPL API v2) is supported —
confirmed with the HSDK5 team, JetPack 7.0–7.1 has no need to be supported by
HSDK5 or its modules ([Slack thread](https://nvidia.enterprise.slack.com/archives/C0B13G3PJLC/p1789085092885369)).
`sipl_compat.hpp` still carries the SIPL API v1 shim inherited from Holoscan
Sensor Bridge, so building against JetPack 7.0–7.1 happens to work, but it is
out of scope: untested, unsupported, and not a target for this operator. Note
"v1"/"v2" are names HSB introduced for this shim, not official SIPL or L4T
version identifiers.

The container build's `L4T_VERSION` build-arg (see [`Dockerfile`](../../Dockerfile))
defaults to `39.2.0`, matching the JP7.2.1 target above. Override it
(`--build-arg L4T_VERSION=<ver>`) to pull SIPL headers for a different L4T
release; only the default is exercised by CI.

## SIPLCaptureOp

| Port | Type | Description |
| --- | --- | --- |
| `frame` | `holoscan::schema::ImageT` | RAW10 (Bayer) or NV12 frame |
| `sensor_data` | `SIPLFrameMetadataT` | Per-frame TSC timestamps, sequence, HSB metadata when present |

`capture_timestamp_ns` in the frame header is derived from
`frameCaptureStartTSC` (start of integration), matching the SDK-wide
contract that capture time is always start-of-integration, never readout or
midpoint.

### Constructor

```cpp
explicit SIPLCaptureOp(
    std::shared_ptr<SIPLCaptureService> service,
    std::uint32_t camera_index,
    holoscan::MemoryKind memory_kind = holoscan::MemoryKind::kCudaDevice);
```

`service` is the shared `SIPLCaptureService` managing the camera rig. The
graph retains it as part of the operator factory, and each constructed
operator shares its lifetime. `camera_index` selects the operator's camera
within that rig and is validated during discovery.

`memory_kind` selects where the published frame is allocated: `kHost`, `kPinnedHost`, or
`kCudaDevice` (default). This is a constructor argument because `setup()`
freezes the placement into the frame port contract. Pick the
placement your own consumer needs: a GPU-resident consumer wants the
default `kCudaDevice`, and a host-only consumer (e.g. writing frames to
disk, as in [sipl_frame_saver](../../applications/sipl_frame_saver)) should
request `kHost` or `kPinnedHost` rather than paying its own device-to-host
copy downstream. `kCudaDevice` additionally requires the application to
bind a device via `CompileOptions::deployment.bind_tensor_output_device()`;
the binding must be *absent* for the two host kinds.

```cpp
auto camera = graph.op<SIPLCaptureOp>(
    "camera", service, /*camera_index=*/0U, holoscan::MemoryKind::kHost);
```

Multiple `SIPLCaptureOp` instances can share one `SIPLCaptureService` for a
multi-camera rig (see [sipl_stereo_monitor](../../applications/sipl_stereo_monitor));
the service starts streaming when the first operator arms and tears down
when the last one stops.

## SIPLCaptureService

Owns the NvSIPL pipeline: device discovery, buffer pool allocation, and the
capture thread. Constructed once per rig and shared by every `SIPLCaptureOp`
in it:

```cpp
auto service = std::make_shared<SIPLCaptureService>(
    camera_config,       // SIPL camera configuration name, e.g. "VB1940"
    json_config,          // vendor JSON platform config path (optional for single-sensor)
    raw_output,           // true = RAW10 Bayer, false = NV12/ISP
    capture_queue_depth,  // default 4
    nito_base_path,       // default "/var/nvidia/nvcam/settings/sipl"
    timeout_us,           // default 1,000,000
    cuda_device);          // default 0; set for the GPU that will consume frames on multi-GPU (IGX dGPU) systems
```

## Tests

```bash
./holoscan_camera test holoscan-camera --ctest-options "-L unit"      # no hardware needed
SIPL_CAMERA_CONFIG=<name> ./holoscan_camera test holoscan-camera --ctest-options "-L hardware"
```

See downstream applications for full working examples:
[sipl_frame_saver](../../applications/sipl_frame_saver) (single camera,
save to disk) and [sipl_stereo_monitor](../../applications/sipl_stereo_monitor)
(two cameras, one shared service). The `kCudaDevice` + device-binding compile
path, which neither app exercises anymore now that both use `kHost`, is
covered instead by `SIPLCaptureOpCompileTest` in
[tests/sipl_capture_op_test.cpp](tests/sipl_capture_op_test.cpp).

## FAQ

**Wrong `image_bfbs_generated.h` picked up at build time?** Mixing headers
from two different Holoscan SDK installs on the same include path (e.g. a
stale `install/` dir reused across an SDK pull) is unsupported and produces
undefined behavior. Wipe and reinstall the SDK (`./run clear_cache` in
`holoscan-sdk`) rather than trying to fix include ordering downstream.
