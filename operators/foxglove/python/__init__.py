"""
SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
SPDX-License-Identifier: Apache-2.0
"""

import holoscan.core  # Registers Holoscan C++ types before loading the extension.
import holoscan.gxf  # noqa: F401  # Registers GXF entity types used by adapter ports.

try:
    from ._foxglove import (
        FoxgloveBatch,
        FoxgloveBox2D,
        FoxgloveCameraCalibration,
        FoxgloveCompressedVideo,
        FoxgloveCompressedVideoAdapterOp,
        FoxgloveDetectionAdapterOp,
        FoxgloveFrameTransform,
        FoxgloveImage,
        FoxgloveImageAnnotations,
        FoxgloveKeyValue,
        FoxglovePoint2D,
        FoxglovePointCloud,
        FoxglovePointsAnnotation,
        FoxglovePoseAdapterOp,
        FoxglovePublisherOp,
        FoxgloveSegmentationMaskAdapterOp,
        FoxgloveTensorAdapterOp,
        FoxgloveText,
        PackedElementField,
        PointsAnnotationType,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import holohub.foxglove._foxglove. Rebuild the Foxglove "
        "operator with the active Holoscan SDK so the pybind11 ABI matches "
        "the runtime environment."
    ) from exc

try:
    from holoscan.core import io_type_registry

    from ._foxglove import register_types as _register_types

    _register_types(io_type_registry)
except ImportError:
    pass

__all__ = [
    "FoxgloveBatch",
    "FoxgloveBox2D",
    "FoxgloveCameraCalibration",
    "FoxgloveCompressedVideo",
    "FoxgloveCompressedVideoAdapterOp",
    "FoxgloveDetectionAdapterOp",
    "FoxgloveFrameTransform",
    "FoxgloveImage",
    "FoxgloveImageAnnotations",
    "FoxgloveKeyValue",
    "FoxglovePoint2D",
    "FoxglovePointCloud",
    "FoxglovePointsAnnotation",
    "FoxglovePoseAdapterOp",
    "FoxglovePublisherOp",
    "FoxgloveSegmentationMaskAdapterOp",
    "FoxgloveTensorAdapterOp",
    "FoxgloveText",
    "PackedElementField",
    "PointsAnnotationType",
]
