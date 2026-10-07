# Jetson Argus capture sample

This example demonstrates how to use the Holoscan Camera `ArgusCaptureOp` to receive camera frames with the Jetson Argus camera API into a realtime Holoscan application.

The sample builds only for ARM64 targets and requires the Jetson Argus runtime
libraries, including for `--validate`. See the operator's
[build instructions](../../operators/argus_capture_op/README.md#build).

```text
ArgusCaptureOp.frame    -> FrameConsumer (image descriptor and tensor extent)
ArgusCaptureOp.metadata -> MetadataConsumer (source timestamps/settings/loss counters)
```

```bash
./holoscan_camera run holoscan_camera_argus [mode] \
  [--camera 0 --mode 0 --width 1920 --height 1080 --fps 30 \]
  [--format nv12 --buffers 6 --drop oldest --frames 120]
```

Select a resolution/frame rate your camera mode supports. The default output is
GPU memory. The consumers read descriptors and settings without copying image
pixels to the CPU. The run completes after the metadata consumer receives the
requested count; Ctrl-C uses Holoscan SDK-managed shutdown by default. Since ports publish
independently, this count need not equal the image consumer's count.

Check runtime graph compilation without opening a camera:

```bash
./build-argus/applications/holoscan_camera_argus/cpp/holoscan_camera_argus \
  --validate --memory host --format i420
```

`--validate` checks graph compilation, not sensor availability or supported
camera modes. Use `--help` for all options. `--integration-offset-ns` is only for
a sensor-calibrated correction; see the operator's timestamp contract before
using it. Source SOF timestamps are printed independently of that option.

If an older sample binary fails with `RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED`,
update and rebuild the sample. SDK versions with managed shutdown signals require
an early `holoscan::ManagedShutdownSignals` domain before graph compilation can
start CUDA helper threads. The sample establishes that domain before constructing
the graph and keeps it alive through capture teardown. Graph validation alone
does not exercise this runtime admission check.

If early setup instead reports `Argus signal setup: kInvalidArgument`, the
process already has additional threads. This can happen when a linked library
starts threads before `main()`. Select caller-managed signals explicitly:

```bash
./build-argus/applications/holoscan_camera_argus/cpp/holoscan_camera_argus \
  --camera 0 --mode 0 --width 1920 --height 1080 --fps 30 \
  --format nv12 --frames 120 --signal-policy caller
```

`--signal-policy caller` skips the HSDK signal domain and calls the SDK's
caller-managed blocking run. The frame limit still requests normal completion.
SIGINT/SIGTERM retain the process's existing handling; with the default OS
disposition, Ctrl-C terminates the process without HSDK cooperative cleanup.
The default `--signal-policy managed` continues to fail explicitly if its early
thread boundary cannot be established; the sample does not switch policies on
its own. The sample requires an SDK with `ManagedShutdownSignals` and
`RunSignalPolicy` support.

## Inspect a captured frame

Use `--save-frame FILE` to save the first image received by the frame consumer.
The file contains tightly packed raw NV12 or I420 pixels and replaces any existing
file at that path. Use `--memory host` or `--memory pinned` to make the pixels
CPU-readable; the default device placement is rejected with this option.
`--validate` never creates a file.

Run from the repository root (the capture command is one line for easy copying):

```bash
./build-argus/applications/holoscan_camera_argus/cpp/holoscan_camera_argus --camera 0 --mode 0 --width 1920 --height 1080 --fps 30 --format nv12 --frames 120 --signal-policy caller --memory host --save-frame frame.nv12
```

Look for `Saved frame: frame.nv12` and a successful exit. The file contains one
frame, even though capture continues to the metadata frame limit. A 1920x1080
NV12 frame occupies 3,110,400 bytes. An unsuccessful write, or a successful capture
that ends before the frame consumer saves an image, produces a nonzero exit.
The first frame may precede automatic exposure and white-balance convergence.

On Ubuntu (including Jetson), install FFmpeg on the machine where you will inspect
the frame. The `ffmpeg` package also includes `ffplay`:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

For other platforms, see the [FFmpeg download page](https://ffmpeg.org/download.html).

Convert the frame to a PNG that can be opened locally or copied off a headless
Jetson:

```bash
ffmpeg -f rawvideo -pixel_format nv12 -video_size 1920x1080 -i frame.nv12 -frames:v 1 -update 1 frame.png
```

Or inspect the raw frame directly with FFplay on a machine with a display:

```bash
ffplay -f rawvideo -pixel_format nv12 -video_size 1920x1080 frame.nv12
```

For `--format i420`, use `-pixel_format yuv420p` in these commands. Always supply
the captured dimensions and format: raw files do not contain that metadata.
See the [FFmpeg rawvideo documentation](https://ffmpeg.org/ffmpeg-formats.html#rawvideo).
The sample does not establish colorimetry, so this conversion is a visual check,
not a calibrated color measurement.
