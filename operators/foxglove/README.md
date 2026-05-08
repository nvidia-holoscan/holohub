# Foxglove Holoscan operator

`FoxglovePublisherOp` publishes `FoxgloveBatch` messages from a Holoscan graph
to Foxglove over WebSocket and can optionally write the same stream to MCAP.
The batch can carry images, camera calibration, image annotations, point clouds,
compressed video, frame transforms, and key-value state variables. The operator
preserves upstream Holoscan capture timestamps from metadata when present, so
MCAP recordings replay against sensor time rather than publication wall-clock
time.

## Purpose

This operator is intended for observability in Holoscan applications. It gives a
running graph a Foxglove Studio endpoint without requiring ROS, and it keeps the
Foxglove data model close to the data already moving through a Holoscan
pipeline: video frames, segmentation masks, detections, state values, point
clouds, encoded video, and frame transforms.

The contribution is centered on the reusable operator. The included
`foxglove_endoscopy_tool_tracking` application is a small runnable example that
shows how to connect the operator to an existing HoloHub workflow.

## Requirements

- Holoscan SDK 4.1.0 or newer.
- CUDA-capable x86_64 or aarch64 platform supported by HoloHub.
- Foxglove Studio, either the desktop application or a compatible browser-based
  session, for live visualization.
- Foxglove SDK 0.23.1. HoloHub fetches this dependency through
  `cmake/FetchFoxgloveSdk.cmake` when building the operator.
- Optional write access to an output directory when `enable_mcap=true`.

## Build and test

Build the operator through the HoloHub CLI:

```bash
./holohub build foxglove
```

Run the C++ unit tests with:

```bash
./holohub test foxglove
```

Python bindings are built when `HOLOHUB_BUILD_PYTHON` is enabled. The bindings
expose the publisher, adapters and Foxglove message carrier types from
`holohub.foxglove`.

Run HoloHub lint before submitting changes:

```bash
./holohub lint operators/foxglove
```

## Quick start

Add the publisher at the end of a graph, start the application, and connect
Foxglove Studio to `ws://localhost:8765`.

For raw frames, use `FoxgloveTensorAdapterOp` or the publisher's direct `image`
port. For inference outputs, use `FoxgloveDetectionAdapterOp` or
`FoxgloveSegmentationMaskAdapterOp`. For remote video monitoring, prefer
`FoxgloveCompressedVideoAdapterOp` behind a Holoscan video encoder so the
WebSocket stream carries H.264 or H.265 rather than full raw frames.

## Architecture

The operator has two layers:

- Adapter operators translate common Holoscan payloads into lightweight
  Foxglove message carrier structs. These adapters handle tensor layout,
  device-to-host staging, timestamp lookup, topic normalization, and message
  construction.
- `FoxglovePublisherOp` owns the Foxglove WebSocket server, optional MCAP
  writer, channel registry, Parameters capability, Services capability, and
  direct modality receivers.

The `messages` input remains the most flexible path and accepts a batch of
already-constructed Foxglove carrier messages. The direct modality ports are
provided for simple graph wiring when an upstream operator already emits a
single image, tensor map, annotation set, point cloud, or key-value state item.

## Ports

### `FoxglovePublisherOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::vector<std::shared_ptr<FoxgloveBatch>>` | Any-size receiver for one or more upstream batches of Foxglove images, compressed video, calibrations, annotations, point clouds, frame transforms and key-value state variables. |
| `image` | `holoscan::gxf::Entity` | Any-size direct image receiver for entities containing `VideoBuffer` or image `Tensor` data. |
| `tensors` | `holoscan::TensorMap` | Any-size direct tensor receiver for image-like tensors, including segmentation masks. |
| `annotations` | `std::shared_ptr<FoxgloveImageAnnotations>` | Any-size direct receiver for image annotations. |
| `point_cloud` | `std::shared_ptr<FoxglovePointCloud>` | Any-size direct receiver for point clouds. |
| `state` | `std::shared_ptr<FoxgloveKeyValue>` | Any-size direct receiver for key-value state messages. |

### `FoxgloveTensorAdapterOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `input` | `holoscan::gxf::Entity` | Entity containing a `Tensor` or `VideoBuffer`. |

Outputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::shared_ptr<FoxgloveBatch>` | One-image batch suitable for `FoxglovePublisherOp`. |

### `FoxgloveDetectionAdapterOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `input` | `holoscan::TensorMap` | `InferenceOp` output tensors containing detection boxes, scores and labels. |

Outputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::shared_ptr<FoxgloveBatch>` | `ImageAnnotations` batch suitable for `FoxglovePublisherOp`. |

### `FoxgloveSegmentationMaskAdapterOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `input` | `holoscan::TensorMap` | Segmentation label tensor, including `SegmentationPostprocessorOp`'s `out_tensor`. |

Outputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::shared_ptr<FoxgloveBatch>` | RawImage mask batch suitable for `FoxglovePublisherOp`. |

### `FoxgloveCompressedVideoAdapterOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `input` | `holoscan::gxf::Entity` | Entity containing an encoded video `Tensor`, including H.264/H.265 output from Holoscan video encoder operators. |

Outputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::shared_ptr<FoxgloveBatch>` | `CompressedVideo` batch suitable for `FoxglovePublisherOp`. |

### `FoxglovePoseAdapterOp` ports

Inputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `input` | `holoscan::TensorMap` | Pose tensor containing either a row-major 4x4 transform matrix or seven `xyz+quat` values. |

Outputs:

| Port | Type | Description |
| ---- | ---- | ----------- |
| `messages` | `std::shared_ptr<FoxgloveBatch>` | `FrameTransform` batch suitable for `FoxglovePublisherOp`. |

## Parameters

### `FoxglovePublisherOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `bind_address` | `std::string` | `127.0.0.1` | WebSocket bind address. Use `0.0.0.0` only when remote clients should be able to connect. |
| `port` | `uint16_t` | `8765` | WebSocket port. |
| `server_name` | `std::string` | `Holoscan Foxglove` | Name advertised to Foxglove. |
| `publish_server_time` | `bool` | `true` | Broadcast latest batch timestamp with Foxglove's Time capability. |
| `drop_when_unsubscribed` | `bool` | `true` | Skip log calls when no WebSocket client or MCAP writer is subscribed. |
| `enable_mcap` | `bool` | `false` | Record to MCAP. |
| `mcap_path` | `std::string` | `holoscan_foxglove.mcap` | MCAP output path. |
| `mcap_compression` | `std::string` | `zstd` | `zstd`, `lz4` or `none`. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `mutable_parameters` | `std::string` | empty | Comma-separated Foxglove parameter names, such as `postprocessor.score_threshold`, that may be changed from Studio. Parameters are read-only unless listed here. |
| `image_topic` | `std::string` | `/image` | Topic used by the direct `image` and `tensors` ports. |
| `image_frame_id` | `std::string` | `camera` | Frame ID used by the direct `image` and `tensors` ports. |
| `image_tensor_name` | `std::string` | empty | Tensor name for the direct `image` and `tensors` ports; empty selects the first tensor. |
| `image_encoding` | `std::string` | empty | RawImage encoding for direct images; empty infers common formats. |
| `image_width` | `uint32_t` | `0` | Width override for direct images. |
| `image_height` | `uint32_t` | `0` | Height override for direct images. |
| `image_step` | `uint32_t` | `0` | Row stride override for direct images. |
| `image_prefer_video_buffer` | `bool` | `true` | Prefer `VideoBuffer` over `Tensor` on the direct `image` port. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

### `FoxgloveTensorAdapterOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `topic` | `std::string` | `/image` | Foxglove topic. |
| `frame_id` | `std::string` | `camera` | Foxglove frame ID. |
| `tensor_name` | `std::string` | empty | Tensor component name; empty selects the first tensor. |
| `encoding` | `std::string` | empty | Foxglove `RawImage` encoding; empty infers common formats. |
| `width` | `uint32_t` | `0` | Width override. |
| `height` | `uint32_t` | `0` | Height override. |
| `step` | `uint32_t` | `0` | Row stride override. |
| `prefer_video_buffer` | `bool` | `true` | Use `VideoBuffer` before tensor when present. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

### `FoxgloveDetectionAdapterOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `annotation_topic` | `std::string` | `/detections` | Foxglove `ImageAnnotations` topic. |
| `boxes_tensor` | `std::string` | `boxes` | Nx4 boxes tensor for separate tensor layouts. |
| `scores_tensor` | `std::string` | `scores` | Optional N-element score tensor. |
| `labels_tensor` | `std::string` | `labels` | Optional N-element class ID tensor. |
| `combined_tensor` | `std::string` | empty | Optional combined detection tensor. |
| `combined_format` | `std::string` | `xyxy_score_label` | `xyxy_score_label` or `batch_label_score_xyxy`. |
| `box_format` | `std::string` | `xyxy` | `xyxy` or `xywh` for separate boxes tensors. |
| `label_map` | `std::string` | empty | Comma-separated labels indexed by class ID. |
| `image_width` | `uint32_t` | `0` | Source image width for normalized coordinates. |
| `image_height` | `uint32_t` | `0` | Source image height for normalized coordinates. |
| `score_threshold` | `double` | `0.25` | Minimum confidence score. |
| `normalized_coordinates` | `bool` | `false` | Scale coordinates from `[0, 1]` into pixels. |
| `clamp_to_image` | `bool` | `true` | Clamp boxes to image bounds when dimensions are known. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

### `FoxgloveSegmentationMaskAdapterOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `topic` | `std::string` | `/segmentation` | Foxglove `RawImage` topic. |
| `frame_id` | `std::string` | `camera` | Foxglove frame ID. |
| `tensor_name` | `std::string` | `out_tensor` | Tensor component name; empty selects the first tensor. |
| `encoding` | `std::string` | `mono8` | Foxglove image encoding for the mask. |
| `width` | `uint32_t` | `0` | Width override. |
| `height` | `uint32_t` | `0` | Height override. |
| `step` | `uint32_t` | `0` | Row stride override. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

### `FoxgloveCompressedVideoAdapterOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `topic` | `std::string` | `/video/compressed` | Foxglove `CompressedVideo` topic. |
| `frame_id` | `std::string` | `camera` | Foxglove frame ID. |
| `tensor_name` | `std::string` | empty | Encoded video tensor component name; empty selects the first tensor. |
| `format` | `std::string` | `h264` | Foxglove compressed video format, commonly `h264` or `h265`. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

### `FoxglovePoseAdapterOp` parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `topic` | `std::string` | `/tf` | Foxglove `FrameTransform` topic. |
| `tensor_name` | `std::string` | empty | Pose tensor name; empty selects the first tensor. |
| `parent_frame_id` | `std::string` | `world` | Parent coordinate frame. |
| `child_frame_id` | `std::string` | `sensor` | Child coordinate frame. |
| `format` | `std::string` | `matrix4x4` | Pose tensor format: `matrix4x4` or `xyz_quat`. |
| `timestamp_metadata_keys` | `std::string` | `acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns` | Metadata keys checked before falling back to Holoscan acquisition timestamp or current time. |
| `allocator` | `std::shared_ptr<holoscan::Allocator>` | `nullptr` | Optional allocator used for pinned host staging buffers during device-to-host copies. |

## C++ example

```cpp
auto adapter = make_operator<ops::FoxgloveTensorAdapterOp>(
    "segmentation_to_foxglove",
    Arg("topic", std::string("/model/segmentation")),
    Arg("encoding", std::string("mono8")),
    Arg("frame_id", std::string("camera")));

auto foxglove = make_operator<ops::FoxglovePublisherOp>(
    "foxglove",
    Arg("bind_address", std::string("127.0.0.1")),
    Arg("port", uint16_t{8765}));

add_flow(segmentation_postprocessor, adapter, {{"out_tensor", "input"}});
add_flow(adapter, foxglove, {{"messages", "messages"}});
```

## Inference examples

Detection tensors from `InferenceOp` can be published as Foxglove annotations:

```cpp
auto detections = make_operator<ops::FoxgloveDetectionAdapterOp>(
    "detections_to_foxglove",
    Arg("annotation_topic", std::string("/detections")),
    Arg("boxes_tensor", std::string("boxes")),
    Arg("scores_tensor", std::string("scores")),
    Arg("labels_tensor", std::string("labels")),
    Arg("label_map", std::string("person,instrument")),
    Arg("image_width", 640u),
    Arg("image_height", 480u),
    Arg("normalized_coordinates", true));

add_flow(inference, detections, {{"transmitter", "input"}});
add_flow(detections, foxglove, {{"messages", "messages"}});
```

`SegmentationPostprocessorOp` emits a device-resident uint8 tensor named
`out_tensor` with shape `(H, W, 1)`. Publish it as a segmentation mask with:

```cpp
auto mask = make_operator<ops::FoxgloveSegmentationMaskAdapterOp>(
    "segmentation_to_foxglove",
    Arg("topic", std::string("/segmentation")),
    Arg("frame_id", std::string("camera")));

add_flow(segmentation_postprocessor, mask, {{"out_tensor", "input"}});
add_flow(mask, foxglove, {{"messages", "messages"}});
```

The publisher also accepts image-like tensors directly when the adapter would
only pass through one modality:

```cpp
auto foxglove = make_operator<ops::FoxglovePublisherOp>(
    "foxglove",
    Arg("image_topic", std::string("/segmentation")),
    Arg("image_encoding", std::string("mono8")));

add_flow(segmentation_postprocessor, foxglove, {{"out_tensor", "tensors"}});
```

## Video and transform examples

Encoded H.264/H.265 tensors can be forwarded as Foxglove compressed video. This
is the preferred path for remote viewing because a 1080p RGBA `RawImage` is
about 8 MB per frame before WebSocket framing.

```cpp
auto compressed_video = make_operator<ops::FoxgloveCompressedVideoAdapterOp>(
    "encoded_video_to_foxglove",
    Arg("topic", std::string("/video/compressed")),
    Arg("format", std::string("h264")),
    Arg("frame_id", std::string("endoscope")));

add_flow(video_encoder, compressed_video, {{"output", "input"}});
add_flow(compressed_video, foxglove, {{"messages", "messages"}});
```

Pose tensors can be published as Foxglove `FrameTransform` messages for the 3D
panel:

```cpp
auto pose = make_operator<ops::FoxglovePoseAdapterOp>(
    "camera_pose_to_foxglove",
    Arg("topic", std::string("/tf")),
    Arg("parent_frame_id", std::string("world")),
    Arg("child_frame_id", std::string("endoscope")),
    Arg("format", std::string("matrix4x4")));

add_flow(pose_source, pose, {{"transforms", "input"}});
add_flow(pose, foxglove, {{"messages", "messages"}});
```

## Performance notes

Foxglove WebSocket and MCAP messages are host-serialized. Device-resident GXF
tensors and video buffers therefore require one device-to-host copy before
publication. The adapters stage those copies through pinned host memory, either
from the supplied `allocator` parameter or a self-owned reusable pinned buffer
pool sized from the first frame. Keep GPU work upstream, publish the narrowest
representation that still preserves debugging value, and prefer compressed
video or `mono8` segmentation masks over RGBA overlays unless the overlay
itself is the artifact under inspection.

When a graph already produces encoded output from `nv_video_encoder` or another
encoder operator, publish that stream as `CompressedVideo`. A 1920x1080 RGBA
`RawImage` is roughly 8 MB per frame before protocol overhead; compressed video
keeps Foxglove usable over a remote connection and avoids turning observability
into the dominant graph cost.

## Timestamp metadata

Adapters look for `acquisition_timestamp_ns`, `timestamp_ns` and
`sensor_timestamp_ns` in Holoscan metadata before falling back to the input
acquisition timestamp and finally current host time. `frame_index` and
`sequence_id` metadata are forwarded as same-timestamp key-value state messages
on `/metadata`.

Override `timestamp_metadata_keys` when a source operator uses a site-specific
metadata key. The first non-zero unsigned timestamp found in that comma-separated
list is used for the Foxglove message timestamp, MCAP `log_time`, server time,
and metadata state messages.

## Foxglove parameters and services

The WebSocket server advertises Foxglove `Time`, `Parameters` and `Services`
capabilities. The Parameters panel lists supported scalar Holoscan parameters
as `<operator>.<parameter>`. Parameters are read-only unless their fully
qualified names are included in `mutable_parameters`.

The operator registers three services:

| Service | Behavior |
| ------- | -------- |
| `start_recording` | Starts MCAP recording at `mcap_path` if it is not already active. |
| `stop_recording` | Flushes and closes the active MCAP writer. |
| `snapshot_mcap` | Closes the current writer and starts a new timestamped MCAP file. |

## Python example

```python
from holohub.foxglove import (
    FoxgloveCompressedVideoAdapterOp,
    FoxglovePoseAdapterOp,
    FoxglovePublisherOp,
    FoxgloveTensorAdapterOp,
)

adapter = FoxgloveTensorAdapterOp(
    self,
    name="video_to_foxglove",
    topic="/video",
    frame_id="camera",
)
publisher = FoxglovePublisherOp(self, name="foxglove", port=8765)

self.add_flow(source, adapter, {("output", "input")})
self.add_flow(adapter, publisher, {("messages", "messages")})
```

## Example application

The `applications/foxglove_endoscopy_tool_tracking` example replays the standard
HoloHub endoscopy sample, runs the existing tool-tracking inference path, and
publishes `/video`, `/detections`, and `/state/inference_fps` to Foxglove
Studio.

```bash
./holohub run foxglove_endoscopy_tool_tracking --language cpp
```

The example does not synthesize pose data because the source dataset does not
carry a camera or instrument transform. Pose publishing is supported by the
operator through `FoxglovePoseAdapterOp`; wire it to a graph that produces a 4x4
transform tensor or `xyz+quat` tensor to publish `/tf`.

## Troubleshooting

If Foxglove Studio connects but the topic list is empty, confirm the application
is actively receiving frames and that `drop_when_unsubscribed` is not hiding
logs while no client is connected. The publisher pre-creates the common channels
on startup, but channels for adapter-created topics appear after the first
message on that topic.

If MCAP output is empty, check `enable_mcap`, `mcap_path`, and filesystem write
permissions. Service calls from Foxglove Studio can start and stop recording at
runtime, so the configured path must also be valid after the graph has started.

If image panels show no overlays, verify that image and annotation topics use
the same timestamp lineage and that annotation coordinates are in image pixels.
Detection adapters can scale normalized coordinates when `image_width`,
`image_height`, and `normalized_coordinates=true` are set.

If a device-resident tensor fails to publish, check that its DLPack dtype and
shape match the adapter. Raw image adapters infer common `mono8`, `rgb8`,
`bgr8`, `rgba8`, and `bgra8` layouts; unusual layouts should set `encoding`,
`width`, `height`, and `step` explicitly.
