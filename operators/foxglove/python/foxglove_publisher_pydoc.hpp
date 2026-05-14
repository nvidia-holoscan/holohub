/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::doc {
namespace FoxglovePublisherOp {
constexpr const char* doc = R"doc(
Publishes FoxgloveBatch messages to Foxglove over a WebSocket endpoint and, optionally, to MCAP.

Example:
    publisher = FoxglovePublisherOp(
        self,
        name="foxglove",
        port=8765,
        enable_mcap=True,
        mcap_path="holoscan_foxglove.mcap",
    )

The publisher preserves upstream Holoscan timestamps when metadata keys such as
``acquisition_timestamp_ns`` are present, and falls back to the current host
time otherwise. It can consume the ``messages`` batch port or direct image,
tensor, annotation, point-cloud and state ports, and advertises Foxglove
Parameters plus MCAP recording services.
)doc";
}  // namespace FoxglovePublisherOp

namespace FoxgloveTensorAdapterOp {
constexpr const char* doc = R"doc(
Converts a Holoscan GXF Entity containing a Tensor or VideoBuffer into a FoxgloveBatch RawImage.

Example:
    adapter = FoxgloveTensorAdapterOp(
        self,
        name="video_to_foxglove",
        topic="/video",
        frame_id="camera",
    )
)doc";
}  // namespace FoxgloveTensorAdapterOp

namespace FoxgloveDetectionAdapterOp {
constexpr const char* doc = R"doc(
Converts InferenceOp detection tensors into Foxglove ImageAnnotations.

Example:
    detections = FoxgloveDetectionAdapterOp(
        self,
        name="detections_to_foxglove",
        annotation_topic="/detections",
        boxes_tensor="boxes",
        scores_tensor="scores",
        labels_tensor="labels",
        label_map="tool,needle",
    )
)doc";
}  // namespace FoxgloveDetectionAdapterOp

namespace FoxgloveSegmentationMaskAdapterOp {
constexpr const char* doc = R"doc(
Converts SegmentationPostprocessorOp label tensors into Foxglove RawImage masks.

Example:
    mask = FoxgloveSegmentationMaskAdapterOp(
        self,
        name="mask_to_foxglove",
        topic="/segmentation",
        tensor_name="out_tensor",
        encoding="mono8",
    )
)doc";
}  // namespace FoxgloveSegmentationMaskAdapterOp

namespace FoxgloveCompressedVideoAdapterOp {
constexpr const char* doc = R"doc(
Converts encoded H.264/H.265 tensors from Holoscan video encoder operators into Foxglove CompressedVideo.

Example:
    compressed = FoxgloveCompressedVideoAdapterOp(
        self,
        name="encoded_video_to_foxglove",
        topic="/video/compressed",
        format="h264",
    )
)doc";
}  // namespace FoxgloveCompressedVideoAdapterOp

namespace FoxglovePoseAdapterOp {
constexpr const char* doc = R"doc(
Converts a 4x4 transform tensor or xyz+quaternion tensor into Foxglove FrameTransform.

Example:
    pose = FoxglovePoseAdapterOp(
        self,
        name="pose_to_foxglove",
        topic="/tf",
        parent_frame_id="world",
        child_frame_id="camera",
        format="matrix4x4",
    )
)doc";
}  // namespace FoxglovePoseAdapterOp
}  // namespace holoscan::doc
