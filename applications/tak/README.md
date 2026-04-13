# Multi-Object Tracking and Detection with TAK Integration

Real-time multi-object detection and tracking with [Team Awareness Kit (TAK)](https://tak.gov/) server integration. Detected objects are visualized locally via Holoviz and uploaded as Cursor-on-Target (CoT) markers to a TAK server for shared situational awareness.

## Overview

This application runs YOLOv8 object detection with ByteTrack multi-object tracking on a live video stream. Each tracked detection is sent as a CoT marker to a TAK server, where it appears on the shared common operating picture (COP) alongside other TAK clients.

The pipeline supports two input modes:

- **V4L2 camera** (default) — any Video4Linux2 compatible device such as a USB webcam
- **Video replayer** — a pre-recorded video converted to GXF entities

### Pipeline

```text
Video Source ──> Format Converter ──> YOLOv8 + ByteTrack ──┬──> Holoviz (local display)
                                                           └──> TAK CoT Operator ──> TAK Server
```

1. **Video Source**: Captures frames from a V4L2 device or replays from a pre-recorded GXF file.
2. **Format Converter**: Resizes frames to 640x640 for the detector.
3. **DetectorOp**: Runs YOLOv8 inference with ByteTrack tracking. Outputs bounding boxes, class labels, and track IDs.
4. **HolovizOp**: Renders the original video with detection overlays (bounding boxes and labels).
5. **TakCotOp**: Converts detections to CoT XML messages and sends them over TCP to a TAK server.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Holoscan SDK**: 3.5.0+
- **V4L2 camera** (optional): USB webcam or compatible device at `/dev/video0`
- **TAK server** (local only): required when running outside Docker

## Model

This application uses [YOLOv8s](https://docs.ultralytics.com/models/yolov8/) from Ultralytics for object detection, paired with [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking. The model weights are downloaded automatically when the application is built.

## Data

A sample traffic video is downloaded from [Pexels](https://www.pexels.com/) when the application is built for use with replayer mode. Please review the [Pexels license terms](https://www.pexels.com/license/).

> **Note:** The user is responsible for checking if the dataset license is fit for the intended purpose.

## TAK Server

The TAK CoT operator sends detection markers to a TAK server over a raw TCP connection using the CoT XML protocol. It is compatible with any TAK server that accepts CoT over TCP, including:

- [OpenTAKServer](https://github.com/brian7704/OpenTAKServer)
- [FreeTAKServer](https://github.com/FreeTAKTeam/FreeTakServer)
- TAK Server (official)

### Built-in OpenTAKServer (Docker only)

When running via the provided Dockerfile, [OpenTAKServer](https://github.com/brian7704/OpenTAKServer) (OTS) is automatically downloaded and installed on the first launch. This provides a fully self-contained demo environment with:

- **TCP CoT endpoint** on port `18088`
- **HTTP API** on port `18081`
- **Web UI** on port `18080` (served by nginx)

The first launch takes 1-2 minutes longer while OTS is downloaded from PyPI and configured. Subsequent launches start immediately.

The default OTS admin credentials are:

- **Username:** `administrator`
- **Password:** `password`

> [!NOTE]
> OpenTAKServer is licensed under [GPL-3.0](https://github.com/brian7704/OpenTAKServer/blob/master/LICENSE). It is **not** included in the container image — it is downloaded directly from its upstream sources (PyPI and GitHub) at runtime. By launching this application, you are obtaining GPL-3.0-licensed software and agree to its license terms. When running outside of Docker, you must provide your own TAK server.

### Configuration

TAK connection settings are configured in `tak.yaml` under the `tak_cot` section:

```yaml
tak_cot:
  tak_host: "localhost"      # TAK server hostname or IP
  tak_port: 18088            # TCP CoT port
  base_lat: 28.54770         # Base latitude for marker placement
  base_lon: -81.37942        # Base longitude for marker placement
```

The `TAK_HOST` environment variable overrides `tak_host` from the config file. Set it to an empty string to disable TAK integration entirely (detections will still be visualized locally).

## Run Instructions

### Docker (recommended)

Build and run the application using the HoloHub CLI:

```bash
./holohub run tak
```

This builds the Docker image and launches the application with a V4L2 camera (default). On first run, OpenTAKServer is downloaded and configured automatically. The OTS Web UI is accessible at `http://localhost:18080` once the container is running.

To use the pre-recorded video replayer instead:

```bash
./holohub run tak replayer
```

### Local (without Docker)

Build the application and download data, then run with a V4L2 camera:

```bash
./holohub run tak --local
```

To use the pre-recorded video replayer instead:

```bash
./holohub run tak replayer --local
```

> [!NOTE]
> When running locally, you must have a TAK server running separately and set `TAK_HOST` accordingly. OpenTAKServer is only started automatically inside the Docker container.

### Custom Video

To convert your own video for use with the replayer, use the provided helper script:

```bash
applications/tak/prepare_video_gxf.sh <input_video.mp4> <output_directory>
```

This handles pixel aspect ratio correction, optional letterboxing, and conversion to GXF entities. Set `TARGET_WIDTH` and `TARGET_HEIGHT` environment variables to enable letterboxing to a specific resolution.

> **Note:** The helper script outputs files named `video_stream.gxf_*`, while `tak.yaml` defaults to `replayer.basename: "traffic"`. After conversion, either rename the output files to `traffic.gxf_index` / `traffic.gxf_entities`, or update `replayer.basename` in `tak.yaml` to `"video_stream"`. Alternatively, use the `--video_dir` flag to point to the output directory and update the basename accordingly.

## Configuration Reference

### `tak.yaml`

| Section | Parameter | Description |
| --- | --- | --- |
| `replayer` | `basename` | Base filename of the GXF video files |
| `replayer` | `repeat` | Loop the video (`true`/`false`) |
| `v4l2_source` | `device` | V4L2 device path (e.g., `/dev/video0`) |
| `detection_preprocessor` | `resize_width/height` | Input resolution for the detector |
| `detector` | `confidence` | Detection confidence threshold |
| `detector` | `model_path` | Path to YOLOv8 `.pt` weights file |
| `tak_cot` | `tak_host` | TAK server hostname |
| `tak_cot` | `tak_port` | TAK server TCP CoT port |
| `tak_cot` | `base_lat/base_lon` | Geographic center for marker placement |
| `tak_cot` | `update_interval` | Minimum seconds between CoT uploads (default: 2.0) |

### `bytetrack.yaml`

| Parameter | Description |
| --- | --- |
| `track_high_thresh` | High detection threshold for track initialization |
| `track_low_thresh` | Low detection threshold for second association |
| `new_track_thresh` | Threshold for creating new tracks |
| `track_buffer` | Number of frames to keep lost tracks |
| `match_thresh` | IoU threshold for matching detections to tracks |

### Environment Variables

| Variable | Description |
| --- | --- |
| `TAK_HOST` | Override TAK server host (empty string disables TAK) |
| `HOLOSCAN_LOG_LEVEL` | Log verbosity: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `HOLOHUB_DATA_PATH` | Override default data directory |
