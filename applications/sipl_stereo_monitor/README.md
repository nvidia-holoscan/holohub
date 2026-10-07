# sipl_stereo_monitor

Sample utility application to validate capture streams across a stereo camera rig.

Drives two [`SIPLCaptureOp`](../../operators/sipl_capture_op) instances
against one shared `SIPLCaptureService`, reporting per-camera frame counts,
cross-camera timestamp skew, and stereo-sync stalls. Use this utility to validate
basic capture streaming functionality with a stereo camera pair on a
SIPL-supported platform.

Both cameras default to host-memory output. `--gpu` selects CUDA-device output
to exercise that capture path. The monitor reads frame headers in either mode;
it does not inspect pixels. Neither mode requires a display or writes images to disk.

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
  --cuda-device N          CUDA device ordinal (default: 0)
  --frames N               Frames per camera before stopping cleanly (default: 500)
  --stall-timeout-s SEC    Idle seconds on either camera (after its first frame) before declaring a stall
  --startup-timeout-s SEC  Seconds to wait for each camera's first frame before declaring a startup failure
  --overall-timeout-s SEC  Hard cap on total run time (default: 120)
  --isp                    Request NV12/ISP output instead of RAW10 (raw_output=false)
  --gpu                    Copy frames to CUDA device memory (default: host memory)
  --debug-pairing          Log every skew pairing match/insert/eviction to stderr
  -h, --help               Show this help text
```

## Run

```bash
./holoscan_camera run sipl_stereo_monitor --run-args \
  '--camera-config VB1940 --json-config applications/config/sipl/vb1940_stereo.json --frames 300'
```

## Check synchronized ISP capture

This runs directly in holoscan-camera; no Isaac OS application is needed.
Use a supported SDK/BSP environment with SIPL headers, driver libraries, the
camera's UDDF driver and NITO files available. Adapt the JSON's network settings
to your board. Configure and build:

```bash
cmake -S . -B build-sipl -DBUILD_ALL=OFF -DAPP_sipl_stereo_monitor=ON \
  -Dholoscan_DIR=/opt/nvidia/holoscan5/lib/cmake/holoscan \
  -DL4T_MAJ_VER=39 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-sipl -j 6
```

Record kernel errors in another terminal, noting the start time of each run:

```bash
sudo journalctl -k -f --since now -o short-iso | tee /tmp/sipl-monitor-kernel.log
```

Run one camera application at a time:

```bash
build-sipl/applications/sipl_stereo_monitor/cpp/sipl_stereo_monitor \
  --camera-config VB1940 \
  --json-config applications/config/sipl/vb1940_stereo.json \
  --isp --gpu --frames 1200 --overall-timeout-s 60
```

Omit `--gpu` for host output. Check for
`reached --frames target on both cameras`, `Timeout getting ISP buffer` warnings, and kernel
`EOF ERR` / `CoE capture status failed` messages. Exit status alone is not enough:
capture errors can occur while frames continue arriving, and the monitor currently
also returns zero on its overall timeout.

Set the camera MAC address in the JSON config to match your connected rig
before running. The CLI launches the application from the repository root, so
relative `--json-config` paths resolve from that directory. For container runs,
the application metadata mounts the host's `/proc/device-tree` read-only;
NvSIPL needs its `model` entry when loading the camera query database. The
`./holoscan_camera` wrapper selects the CLI version automatically. If you
override the mode's Docker options with `--docker-opts`, retain the read-only
`/proc/device-tree` mount.
