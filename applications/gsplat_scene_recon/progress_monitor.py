# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
G-SHARP Pipeline Progress Monitor

Displays real-time progress bars for batch pipeline stages (VGGT, format
conversion, GSplat training) by reading a shared progress JSON file and
rendering status via HoloViz rectangles and text overlays.

Usage (inside Docker with display forwarding):
    python progress_monitor.py --progress-file /path/to/progress.json

The progress file is written by each pipeline stage using the
``stages.progress.update_progress()`` utility.
"""

from __future__ import annotations

import time
from argparse import ArgumentParser

import cupy as cp
import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator
from stages.progress import read_progress

# Layout constants (normalised [0, 1] coordinates)
BAR_LEFT = 0.10
BAR_RIGHT = 0.85
BAR_HEIGHT = 0.035
BAR_GAP = 0.14
TOP_Y = 0.22

STAGES = [
    {"key": "vggt", "label": "Phase 2: VGGT Pose Estimation"},
    {"key": "format_conversion", "label": "Phase 3: Format Conversion"},
    {"key": "training", "label": "Phase 4: GSplat Training"},
]

# Colours (RGBA)
COLOR_BG = [0.25, 0.25, 0.30, 1.0]
COLOR_FILL_RUNNING = [0.15, 0.65, 0.95, 1.0]
COLOR_FILL_COMPLETE = [0.20, 0.80, 0.35, 1.0]
COLOR_FILL_ERROR = [0.90, 0.25, 0.20, 1.0]
COLOR_TEXT = [1.0, 1.0, 1.0, 1.0]
COLOR_DIM_TEXT = [0.55, 0.55, 0.60, 1.0]
COLOR_TITLE = [0.95, 0.95, 0.95, 1.0]


class ProgressMonitorOp(Operator):
    """Reads progress.json and emits HoloViz primitives for progress bars."""

    def __init__(self, fragment, progress_file, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._progress_file = progress_file
        self._last_poll = 0.0
        self._poll_interval = 0.4
        self._cached_data = None

    def setup(self, spec: OperatorSpec):
        spec.output("out_tensors")
        spec.output("out_specs")

    def compute(self, op_input, op_output, context):
        now = time.monotonic()
        if now - self._last_poll >= self._poll_interval:
            data = read_progress(self._progress_file)
            if data is not None:
                self._cached_data = data
            self._last_poll = now

        tensors, specs = self._build_visuals(self._cached_data)
        op_output.emit(tensors, "out_tensors")
        op_output.emit(specs, "out_specs")

        time.sleep(0.033)

    def _build_visuals(self, data: dict | None):
        tensors = {}
        specs = []

        # Title text
        tensors["title_pos"] = cp.asarray(np.array([[0.5, 0.08]], dtype=np.float32))
        spec = HolovizOp.InputSpec("title_pos", "text")
        spec.text = ["G-SHARP Pipeline Progress"]
        spec.color = COLOR_TITLE
        spec.priority = 10
        specs.append(spec)

        # The JSON now has {"active": "stage_key", "stages": {key: {...}, ...}}
        stages_data = data.get("stages", {}) if data else {}

        for idx, stage_info in enumerate(STAGES):
            y_top = TOP_Y + idx * BAR_GAP
            y_bot = y_top + BAR_HEIGHT
            stage_key = stage_info["key"]
            label = stage_info["label"]

            sdata = stages_data.get(stage_key)
            if sdata is not None:
                current = sdata.get("current", 0)
                total = max(sdata.get("total", 1), 1)
                detail = sdata.get("detail", "")
                status = sdata.get("status", "running")
                progress = min(current / total, 1.0)
            else:
                progress = 0.0
                detail = "Pending"
                status = "pending"

            # Stage label
            lbl_name = f"label_{idx}"
            tensors[lbl_name] = cp.asarray(np.array([[BAR_LEFT, y_top - 0.045]], dtype=np.float32))
            spec = HolovizOp.InputSpec(lbl_name, "text")
            if status == "pending":
                spec.text = [f"{label}  [pending]"]
                spec.color = COLOR_DIM_TEXT
            else:
                pct = int(progress * 100)
                spec.text = [f"{label}  {pct}%"]
                spec.color = COLOR_TEXT
            spec.priority = 10
            specs.append(spec)

            # Background bar
            bg_name = f"bar_bg_{idx}"
            tensors[bg_name] = cp.asarray(
                np.array([[BAR_LEFT, y_top], [BAR_RIGHT, y_bot]], dtype=np.float32)
            )
            spec = HolovizOp.InputSpec(bg_name, "rectangles")
            spec.color = COLOR_BG
            spec.priority = 5
            specs.append(spec)

            # Filled bar
            if progress > 0.001:
                fill_name = f"bar_fill_{idx}"
                fill_right = BAR_LEFT + progress * (BAR_RIGHT - BAR_LEFT)
                tensors[fill_name] = cp.asarray(
                    np.array([[BAR_LEFT, y_top], [fill_right, y_bot]], dtype=np.float32)
                )
                spec = HolovizOp.InputSpec(fill_name, "rectangles")
                if status == "complete":
                    spec.color = COLOR_FILL_COMPLETE
                elif status == "error":
                    spec.color = COLOR_FILL_ERROR
                else:
                    spec.color = COLOR_FILL_RUNNING
                spec.priority = 6
                specs.append(spec)

            # Detail text
            if detail and status != "pending":
                det_name = f"detail_{idx}"
                tensors[det_name] = cp.asarray(
                    np.array([[BAR_LEFT, y_bot + 0.015]], dtype=np.float32)
                )
                spec = HolovizOp.InputSpec(det_name, "text")
                spec.text = [detail]
                spec.color = COLOR_DIM_TEXT
                spec.priority = 10
                specs.append(spec)

        return tensors, specs


class ProgressMonitorApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        monitor = ProgressMonitorOp(
            self,
            progress_file=self._args.progress_file,
            name="monitor",
        )

        pool = UnboundedAllocator(self, name="pool")
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            headless=self._args.headless,
            width=720,
            height=480,
            window_title="G-SHARP Pipeline Progress",
            tensors=[],
        )

        self.add_flow(monitor, holoviz, {("out_tensors", "receivers")})
        self.add_flow(monitor, holoviz, {("out_specs", "input_specs")})


def main():
    parser = ArgumentParser(description="G-SHARP Pipeline Progress Monitor")
    parser.add_argument(
        "--progress-file", required=True, help="Path to the shared progress.json file"
    )
    parser.add_argument("--headless", action="store_true", help="Run without display window")
    args = parser.parse_args()

    app = ProgressMonitorApp(args)
    app.run()


if __name__ == "__main__":
    main()
