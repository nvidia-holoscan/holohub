# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared progress reporting utility for G-SHARP pipeline stages.

Each stage writes a JSON file to a known path so that the HoloViz progress
monitor can display real-time progress bars.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def update_progress(
    path: str | Path,
    stage: str,
    label: str,
    current: int,
    total: int,
    detail: str = "",
    status: str = "running",
) -> None:
    """Write a progress update to a shared JSON file.

    The file stores a ``stages`` dict keyed by stage identifier so that
    completed stages remain visible even after a faster stage finishes
    before the monitor polls again.  An ``active`` field tracks which
    stage most recently wrote an update.

    Uses atomic write (write-to-temp then rename) to avoid partial reads
    by the monitoring process.

    Parameters
    ----------
    path : str or Path
        Path to the shared progress JSON file.
    stage : str
        Stage identifier: "vggt", "format_conversion", or "training".
    label : str
        Human-readable stage name for display.
    current : int
        Current step (0-based or 1-based — just be consistent).
    total : int
        Total number of steps.
    detail : str
        Optional detail string (e.g. "Batch 2/4: frames 30-59").
    status : str
        One of "running", "complete", or "error".
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing data to preserve other stages' state
    existing: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Migrate flat (v1) format to nested format
    if "stages" not in existing:
        existing = {"active": None, "stages": {}}

    existing["active"] = stage
    existing["stages"][stage] = {
        "label": label,
        "current": current,
        "total": total,
        "detail": detail,
        "status": status,
    }

    # Atomic write: temp file in same directory, then rename
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            # Best-effort cleanup: ignore failures deleting the temporary file
            pass
        raise


def read_progress(path: str | Path) -> dict | None:
    """Read the current progress from the shared JSON file.

    Returns the full dict which contains ``active`` (str) and ``stages``
    (dict of stage_key → {label, current, total, detail, status}).
    Returns None if the file does not exist or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
