# SIPL Capture Benchmarks and Qualification Results

## Scope and setup

These results record stereo-sync stability qualification for `SIPLCaptureOp`
0.1.0, tracked in [NVBUG 6714027](https://nvbugspro.nvidia.com/bug/6714027)
("Startup failure rate of synced stereo cameras through SIPL, with HSB
vb1940 UDDF driver"). They cover run-to-run stability of stereo-synchronized
capture (does the stream stay alive and stay in sync) rather than throughput
or per-frame latency.

Hardware: Leopard Imaging VB1940 Eagle stereo pair on AGX Thor, connected
through the HSB UDDF driver over COE (Camera-over-Ethernet). Two tools were
used:

- `sipl_stereo_monitor` (in this module, named `sipl_stereo_debug` at the time
  these runs were recorded) — exercises the full
  `SIPLCaptureOp`/`SIPLCaptureService` stack through HSDK5:

  ```bash
  ./holoscan_camera run sipl_stereo_monitor --as-root --docker-opts \
      "--privileged -v /var/nvidia/nvcam/settings:/var/nvidia/nvcam/settings:ro \
       -v /sys/devices:/sys/devices -v /tmp/argus_socket:/tmp/argus_socket" \
      --run-args "--camera-config VB1940 --json-config <path-to-vb1940_stereo.json> \
                  [--isp] --frames 3000 --stall-timeout-s 6"
  ```

- `nvsipl_camera` — a JetPack-preinstalled tool (`/usr/src/jetson_sipl_api/sipl/`)
  that talks to SIPL directly, with no HSDK5, HSB, or holoscan-camera code in
  the path:

  ```bash
  echo "HsbTransport,0,mgbe0_0,<mac>,<ip>" > coe.csv
  nvsipl_camera -c VB1940_Stereo -H -Z -R -r 35 --disableISP1 --disableISP2 \
      --coeConfigOverridePath coe.csv -m 0x11 -s
  ```

Comparing the two isolates whether a failure originates in this operator or
below it.

## Recorded results

### `sipl_stereo_debug` (through `SIPLCaptureOp`), 30 runs x 300 frames, RAW10

| Metric | Value |
| --- | --- |
| Clean completions (300+ frames) | 27/30 (90%) |
| Stalls | 3/30, all on camera0, at frames 90/118/118 |
| Runs with cam-to-cam skew swinging to ~±18 ms | 13/30 |

The 13 skewed runs kept delivering frames throughout — a sync-quality
symptom, not a liveness one.

### `nvsipl_camera` (bypassing HSDK5/HSB/holoscan-camera entirely), 100 runs each

| Metric | Run set 1 | Run set 2 |
| --- | --- | --- |
| Healthy runs | 81/100 (81%, 95% Wilson CI [72.2, 87.5]) | 88/100 (88%) |
| Runs that never established streaming | 2 | 0 |
| Runs reporting `SUCCESS` but not actually healthy | 8 | 0 |
| Terminal stalls | 6 | 12 |
| Kernel `CoE capture status failed: -110` | 19 | 12 |
| Pooled steady-state frame loss (once streaming) | 0/162494 (0.0000%) | not recorded |

## Interpretation and follow-up

### The failure exists below `SIPLCaptureOp`, at a comparable rate

`nvsipl_camera` reproduces the same class of failure (terminal stalls,
`CoE capture status failed: -110`) with no HSDK5, HSB, or holoscan-camera
code involved, at roughly the same order of magnitude (81-88% healthy) as
`sipl_stereo_debug` through the full operator stack (90% clean). The stall
itself lives inside the acquire thread's wait on SIPL's own completion
queue — `SIPLCaptureOp` is a pure consumer there. This points at
SIPL/firmware/link-layer as the root cause, not this operator; see
NVBUG 6714027 for ongoing investigation and the camera-team discussion.

### Camera firmware version affects the failure rate

Runs against HSB firmware `0x2606` failed consistently on one physical
unit (100% failure) but not on another of the same firmware version,
suggesting a possible unit- or cable-specific factor (QSFP/SFP+ adapter
was also swapped between units). A separate run against older firmware
`0x2507` (known to have startup issues) saw a ~30% failure rate even
*without* stereo sync enabled. Firmware/unit was not held constant across
all recorded runs, so these numbers should not be read as a controlled
firmware comparison — only as evidence that firmware/hardware variance is
a real factor.

### Cam-to-cam skew alternation

The ~±18 ms skew seen in some `sipl_stereo_debug` runs traces back to
`frameCaptureStartTSC` itself, read directly from SIPL's buffer metadata
before any operator/service code touches it — not a measurement artifact
of this module (confirmed by observing the same alternation in a single
camera's own frame-to-frame timestamps, independent of any cross-camera
comparison).

### What to check before relying on stereo sync in production

Consumers of `SIPLCaptureOp` with `sync_sensors: true` should treat
startup as unreliable in the current SIPL/HSB stack: expect an occasional
stall (observed 3-19% depending on run and hardware) and validate stream
health past the first few seconds rather than assuming a clean start
implies a clean run. See [release notes](../RELEASE_NOTES.md) for the
concise known-issue summary and [NVBUG 6714027](https://nvbugspro.nvidia.com/bug/6714027)
for status.
