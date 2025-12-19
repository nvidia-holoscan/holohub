# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("CuPy is required for GPU processing (`pip install cupy-cuda12x`).") from exc

try:
    from pyuff_ustb import Uff
    from pyuff_ustb.readers.base import ReaderKeyError
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("pyuff_ustb is required to read UFF files (`pip install pyuff-ustb`).") from exc


STREAMLIT_MAX_DIM = 65000
UFF_DATASET_CANDIDATES = ("beamformed", "beamformed_data", "b_data")
UFF_PRIMARY_FIELDS = ("data", "b_matrix", "frame_data")


def load_uff_frame(
    path: str | Path,
    dataset: str | None = None,
    frame_index: int = 0,
) -> Dict[str, Any]:
    """Load a single B-mode frame from a UFF file directly into GPU memory."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"UFF file not found: {resolved}")

    reader = Uff(str(resolved))
    payload = None
    chosen_dataset: str | None = None
    candidates = [dataset] if dataset is not None else list(UFF_DATASET_CANDIDATES)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            payload = reader.read(candidate)
        except ReaderKeyError:
            continue
        chosen_dataset = candidate
        break

    if payload is None or chosen_dataset is None:
        available = _available_datasets(reader)
        attempted = candidates if dataset is None else [dataset]
        attempted_str = ", ".join(str(name) for name in attempted if name)
        raise ValueError(
            f"Could not locate a supported dataset in '{resolved.name}'. "
            f"Tried: {attempted_str or '(none)'}; "
            f"available datasets: {available or 'unknown'}"
        )

    data = _extract_numpy_array(payload)
    tensor = cp.asarray(data, order="C")
    tensor = _prepare_tensor(tensor)
    tensor, axes_taken, inferred_shape = _select_frame(
        tensor, payload=payload, frame_index=frame_index
    )
    tensor = _normalize_unit_interval(tensor)

    meta: Dict[str, Any] = {
        "path": str(resolved),
        "dataset": chosen_dataset,
        "frame_index": frame_index,
        "frame_axes": list(axes_taken),
        "image_shape": inferred_shape,
        "shape": tuple(int(dim) for dim in tensor.shape),
        "dtype": str(tensor.dtype),
    }
    meta.update(_extract_metadata(payload))

    return {"data": tensor, "meta": meta}


def _extract_numpy_array(payload: Any) -> Any:
    """Best-effort extraction of the numeric payload from pyuff_ustb output."""

    result = None
    if hasattr(payload, "data"):
        result = payload.data  # type: ignore[attr-defined]
    elif hasattr(payload, "b_matrix"):
        result = payload.b_matrix  # type: ignore[attr-defined]
    elif isinstance(payload, dict):
        for key in UFF_PRIMARY_FIELDS:
            if key in payload:
                result = payload[key]
                break
    if result is None:
        raise ValueError("Could not locate numeric data within the provided UFF dataset.")
    return result


def _prepare_tensor(tensor: cp.ndarray) -> cp.ndarray:
    """Ensure tensor is real-valued float32."""

    if cp.iscomplexobj(tensor):
        tensor = cp.abs(tensor)
    return cp.asarray(tensor, dtype=cp.float32, order="C")


def _available_datasets(reader: Uff) -> list[str]:
    """Return a flat list of available dataset names, if discoverable."""

    names: list[str] = []
    root = getattr(reader, "_reader", None)
    if root is not None and hasattr(root, "file"):
        try:
            import h5py
        except ImportError:
            h5py = None  # type: ignore[assignment]

        filepath = getattr(root.file, "filename", None)
        if h5py is not None and filepath:
            try:
                with h5py.File(filepath, "r") as h5f:
                    h5f.visit(lambda name: names.append(name))
            except OSError:
                names = []
    return sorted(set(names))


def _select_frame(
    tensor: cp.ndarray,
    payload: Any,
    frame_index: int,
) -> Tuple[cp.ndarray, Tuple[int, ...], Optional[Tuple[int, int]]]:
    """Reduce tensor to 2D by iteratively slicing along the smallest axes."""

    if frame_index < 0:
        raise IndexError("Frame index must be non-negative.")

    axes_taken: List[int] = []
    current = tensor

    while current.ndim > 2:
        axis = min(range(current.ndim), key=lambda ax: current.shape[ax])
        if frame_index >= current.shape[axis]:
            raise IndexError(f"Frame index {frame_index} out of bounds for axis {axis} with size {current.shape[axis]}.")
        current = cp.take(current, indices=frame_index, axis=axis)
        axes_taken.append(axis)

    inferred_shape: Optional[Tuple[int, int]] = None
    if current.ndim == 1:
        current, inferred_shape = _reshape_flat_tensor(current, payload)
    elif current.ndim == 2 and min(current.shape) <= 1 and max(current.shape) > STREAMLIT_MAX_DIM:
        flattened = cp.ravel(current)
        current, inferred_shape = _reshape_flat_tensor(flattened, payload)

    return current, tuple(axes_taken), inferred_shape


def _normalize_unit_interval(tensor: cp.ndarray) -> cp.ndarray:
    """Normalize data to the [0, 1] range directly on the GPU."""

    tensor = tensor.astype(cp.float32, copy=False)
    finite_tensor = cp.nan_to_num(tensor, copy=False)
    lo = float(cp.nanmin(finite_tensor))
    hi = float(cp.nanmax(finite_tensor))

    normalized = cp.zeros_like(finite_tensor, dtype=cp.float32)
    if hi != lo:
        normalized = (finite_tensor - lo) / (hi - lo)
        normalized = cp.clip(normalized, 0.0, 1.0)
    return normalized


def _reshape_flat_tensor(
    vector: cp.ndarray,
    payload: Any,
) -> Tuple[cp.ndarray, Optional[Tuple[int, int]]]:
    """Reshape a 1D vector into a 2D image using scan metadata when possible."""

    length = int(vector.shape[0])
    inferred_shape: Optional[Tuple[int, int]] = None
    reshaped = vector
    if length == 0:
        reshaped = vector.reshape((0, 0))
    else:
        inferred = _infer_image_shape(payload, length)
        if inferred:
            reshaped = cp.reshape(vector, inferred, order="F")
            inferred_shape = inferred
        else:
            width = int(math.isqrt(length))
            while width > 1 and length % width != 0:
                width -= 1
            if width > 1:
                height = length // width
                reshaped = cp.reshape(vector, (height, width), order="F")
                inferred_shape = (height, width)
            else:
                max_dim = STREAMLIT_MAX_DIM
                if length > max_dim:
                    vector = vector[:max_dim]
                reshaped = cp.reshape(vector, (1, vector.size), order="C")
                inferred_shape = (1, int(vector.size))

    return cp.ascontiguousarray(reshaped), inferred_shape


def _infer_image_shape(payload: Any, numel: int) -> Optional[Tuple[int, int]]:
    """Infer a 2D image shape from scan metadata."""

    scan = getattr(payload, "scan", None)
    inferred: Optional[Tuple[int, int]] = None
    if scan is not None:
        candidates: List[Tuple[int, int]] = []

        nx = _coerce_int(getattr(scan, "N_x_axis", None))
        nz = _coerce_int(getattr(scan, "N_z_axis", None))
        if nx and nz:
            candidates.append((nz, nx))

        unique_x = _unique_coordinate_count(getattr(scan, "x", None))
        unique_z = _unique_coordinate_count(getattr(scan, "z", None))
        if unique_x and unique_z:
            candidates.append((unique_z, unique_x))

        x_axis = getattr(scan, "x_axis", None)
        z_axis = getattr(scan, "z_axis", None)
        if x_axis is not None and z_axis is not None:
            axis_x_len = _unique_coordinate_count(x_axis)
            axis_z_len = _unique_coordinate_count(z_axis)
            if axis_x_len and axis_z_len:
                candidates.append((axis_z_len, axis_x_len))

        for height, width in candidates:
            if height * width == numel:
                inferred = (int(height), int(width))
                break

    return inferred


def _coerce_int(value: Any) -> Optional[int]:
    intval: Optional[int] = None
    try:
        intval_candidate = int(value)
        if intval_candidate > 0:
            intval = intval_candidate
    except (TypeError, ValueError):
        intval = None
    return intval


def _unique_coordinate_count(axis: Any) -> Optional[int]:
    count: Optional[int] = None
    if axis is not None:
        try:
            arr = cp.asarray(axis)
            unique = cp.unique(arr)
            size = int(unique.size)
            if size > 0:
                count = size
        except Exception:
            count = None
    return count


def _extract_metadata(payload: Any) -> Dict[str, Any]:
    """Extract a small metadata snapshot for UI display."""

    meta: Dict[str, Any] = {}
    if hasattr(payload, "title"):
        meta["title"] = getattr(payload, "title")
    if hasattr(payload, "probe"):
        meta["probe"] = getattr(payload, "probe")
    if hasattr(payload, "sequence"):
        meta["sequence"] = getattr(payload, "sequence")
    return meta


__all__ = ["load_uff_frame"]
