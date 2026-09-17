# sipl_stereo_monitor

Sample utility application to validate capture streams across a stereo camera rig.

Drives two [`SIPLCaptureOp`](../../operators/sipl_capture_op) instances
against one shared `SIPLCaptureService`, reporting per-camera frame counts,
cross-camera timestamp skew, and stereo-sync stalls. Use this utility to validate
basic capture streaming functionality with a stereo camera pair on a
SIPL-supported platform.

Both cameras are constructed with `MemoryKind::kHost`: this tool only reads
frame timestamps and sequence numbers, never the pixel tensor, so there is
no consumer here to justify `SIPLCaptureOp`'s GPU-resident default.

## Requirements

- Requires an NvSIPL supported platform, such as:
  - NVIDIA Jetson AGX Thor with Jetpack >= 7.0
  - NVIDIA IGX Thor Mini with IGX OS >= 2.0
- Requires a stereo VB1940 camera input via Holoscan Sensor Bridge FPGA

## Usage

```bash
sipl_stereo_monitor --camera-config VB1940 --json-config applications/config/sipl/vb1940_stereo.json
```

```text
Options:
  --camera-config NAME     SIPL camera configuration name (default: VB1940)
  --json-config PATH       Vendor JSON platform config (required for this rig)
  --frames N               Frames per camera before stopping cleanly (default: 500)
  --stall-timeout-s SEC    Idle seconds on either camera (after its first frame) before declaring a stall
  --startup-timeout-s SEC  Seconds to wait for each camera's first frame before declaring a startup failure
  --overall-timeout-s SEC  Hard cap on total run time (default: 120)
  --isp                    Request NV12/ISP output instead of RAW10 (raw_output=false)
  --debug-pairing          Log every skew pairing match/insert/eviction to stderr
  -h, --help               Show this help text
```

## Run

```bash
./holoscan_camera run sipl_stereo_monitor
```
