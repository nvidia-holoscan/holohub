# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Minimal Holoscan app using reusable source/pipeline operators (UFF or raysim).

Graph layout: Source -> PipelineOp -> RGBA formatter -> Holoviz/FPS logger.
Assumptions: Holoscan SDK, Holoviz, and CuPy are installed and available.
Zero-copy is intentionally skipped; this uses the standard Holoviz path.

Run with a static UFF frame:
  python -m ultra_post.app.holoscan_app --uff ultra_post/examples/demo.uff

Run with the i4h-sensor-simulation raysim backend:
  python -m ultra_post.app.holoscan_app --source raysim --sim-range -20 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from ultra_post.app.holoscan_operators import (
    FuncOp,
    UffSourceConfig,
    make_raysim_source_op,
    make_rgba_formatter_op,
    make_uff_source_op,
)
from ultra_post.core.pipeline import Pipeline, create_node, pipeline_from_yaml
from ultra_post.filters.registry import FILTERS

if TYPE_CHECKING:
    from ultra_post.sim.raysim_source import RaysimFrameGenerator


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultrasound post-processing via Holoscan")
    p.add_argument(
        "--source",
        choices=("uff", "raysim"),
        default="uff",
        help="Frame source to use (default: uff).",
    )
    DEFAULT_UFF_PATH = os.path.join(
        os.environ.get("HOLOHUB_DATA_PATH", "../data"), "ultrasound_postprocessing/demo.uff"
    )
    p.add_argument(
        "--uff", type=str, default=DEFAULT_UFF_PATH, help="Path to UFF file when --source=uff"
    )
    p.add_argument("--dataset", type=str, default=None, help="Dataset name inside UFF (optional)")
    p.add_argument("--fps", type=float, default=30.0, help="Target frame rate")
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Pipeline preset YAML (optional; default = just Gamma)",
    )
    p.add_argument(
        "--headless", action="store_true", help="Run without display (offscreen Holoviz)"
    )
    p.add_argument(
        "--log",
        type=str,
        default="us_post_processing.log",
        help="Path to write Holoscan flow tracking logs",
    )
    # Raysim-specific knobs (ignored for UFF mode)
    p.add_argument("--sim-frames", type=int, default=60, help="Frames per simulated sweep loop")
    p.add_argument(
        "--sim-range",
        type=float,
        nargs=2,
        metavar=("X_START_MM", "X_END_MM"),
        default=(-20.0, 20.0),
        help="Probe sweep limits along the x-axis in millimeters",
    )
    p.add_argument(
        "--sim-dynamic-range",
        type=float,
        nargs=2,
        metavar=("MIN_DB", "MAX_DB"),
        default=(-60.0, 0.0),
        help="Dynamic range window applied before feeding the filter pipeline",
    )
    p.add_argument(
        "--sim-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(512, 512),
        help="Target (height, width) for simulated B-mode images",
    )
    p.add_argument(
        "--sim-sector",
        type=float,
        default=73.0,
        help="Curvilinear probe sector angle in degrees (raysim mode)",
    )
    p.add_argument(
        "--sim-radius",
        type=float,
        default=45.0,
        help="Curvilinear probe radius in millimeters (raysim mode)",
    )
    p.add_argument(
        "--sim-frequency",
        type=float,
        default=5.0,
        help="Probe center frequency in MHz (raysim mode)",
    )
    p.add_argument(
        "--sim-world",
        type=str,
        default="fat",
        help="Background material name for the raysim world",
    )
    return p.parse_args(argv)


def _load_pipeline_from_yaml(path: Optional[str]) -> Pipeline:
    filters = FILTERS
    if not path:
        pipeline: Pipeline = []
        if filters:
            filter_name = (
                "gamma_compression"
                if "gamma_compression" in filters
                else next(iter(filters.keys()))
            )
            pipeline.append(create_node(filter_name))
        return pipeline

    yaml_text = Path(path).read_text(encoding="utf-8")
    return pipeline_from_yaml(yaml_text, filters=filters)


def _build_raysim_generator(
    sim_frames: int,
    sim_range: Sequence[float],
    sim_dynamic_range: Sequence[float],
    sim_size: Sequence[int],
    sim_sector: float,
    sim_radius: float,
    sim_frequency: float,
    sim_world: str,
) -> "RaysimFrameGenerator":
    from ultra_post.sim.raysim_source import RaysimFrameGenerator, RaysimSweepConfig

    sweep_cfg = RaysimSweepConfig(
        frames_per_loop=int(sim_frames),
        sweep_range_mm=_pair(sim_range, (-20.0, 20.0)),
        dynamic_range_db=_pair(sim_dynamic_range, (-60.0, 0.0)),
        b_mode_size=tuple(int(v) for v in sim_size),
        sector_angle_deg=float(sim_sector),
        radius_mm=float(sim_radius),
        frequency_mhz=float(sim_frequency),
        world_name=str(sim_world),
    )
    return RaysimFrameGenerator(sweep_cfg)


def _pair(values: Sequence[float], default: Tuple[float, float]) -> Tuple[float, float]:
    pair = default
    if len(values) == 2:
        pair = (float(values[0]), float(values[1]))
    return pair


def run_holoscan_app(args: argparse.Namespace) -> None:
    import os
    from pathlib import Path as _Path

    from holoscan.conditions import PeriodicCondition  # type: ignore
    from holoscan.core import Application, Tracker  # type: ignore
    from holoscan.operators import HolovizOp  # type: ignore

    if args.preset is None:
        print("No preset provided. Using default preset")
    pipeline = _load_pipeline_from_yaml(args.preset)
    to_rgba = make_rgba_formatter_op()

    if args.source == "uff":
        if not args.uff:
            raise ValueError("--uff path must be provided when --source=uff")
        source_op = make_uff_source_op(
            UffSourceConfig(path=Path(args.uff), dataset=args.dataset, frame_index=0)
        )
    elif args.source == "raysim":
        generator = _build_raysim_generator(
            args.sim_frames,
            args.sim_range,
            args.sim_dynamic_range,
            args.sim_size,
            args.sim_sector,
            args.sim_radius,
            args.sim_frequency,
            args.sim_world,
        )
        source_op = make_raysim_source_op(generator)
    else:
        raise ValueError(f"Unsupported source '{args.source}'")

    class USPPApp(Application):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()

        def compose(self) -> None:  # noqa: D401
            # Schedule the source periodically via a PeriodicCondition.
            # Use integer nanoseconds for recess_period as required by the API.
            freq_hz = max(1.0, float(args.fps))
            period_ns = int(1e9 / freq_hz)
            source = source_op(
                self,
                PeriodicCondition(self, recess_period=period_ns),
                name="frame_source",
            )
            prev = source

            # Dynamically build the pipeline graph from the config list
            for i, node in enumerate(pipeline):
                if not node.get("enabled", True):
                    continue

                op_name = node["op"]
                params = node.get("params", {})
                func = FILTERS.get(op_name)
                if not func:
                    print(f"Warning: Op '{op_name}' not found, skipping.")
                    continue

                # Instantiate stateful operators when the registry stores classes
                if isinstance(func, type):
                    func = func()

                # Efficiently instantiate the single generic FuncOp class
                step_op = FuncOp(self, name=f"{op_name}_{i}", fn=func, params=params)

                self.add_flow(prev, step_op, {("out", "in")})
                prev = step_op

            fmt = to_rgba(self, name="to_rgba")
            self.add_flow(prev, fmt, {("out", "in")})

            # Always log FPS; attempt Holoviz with offscreen fallback when no display.
            gpu_present = any(
                _Path(p).exists()
                for p in ("/dev/nvidiactl", "/dev/nvidia0", "/proc/driver/nvidia/version")
            )
            display_present = bool(os.environ.get("DISPLAY"))
            headless_mode = bool(args.headless or (not display_present) or (not gpu_present))

            # If an NVIDIA GPU is present, create Holoviz; use offscreen when headless or no display.
            if gpu_present:
                if headless_mode:
                    viz = HolovizOp(self, name="holoviz", headless=True)
                else:
                    viz = HolovizOp(self, name="holoviz")
                self.add_flow(fmt, viz, {("out", "receivers")})

    app = USPPApp()
    # Enable Data Flow Tracking and save logs
    print(f"Initializing Tracker with log file: {args.log}")
    try:
        with Tracker(app, filename=args.log) as tracker:  # noqa: F841
            print("Tracker started. Running app...")
            app.run()
            print("App run finished.")
    except KeyboardInterrupt:
        print("User interrupted.")
    except Exception as e:
        print(f"Error during run: {e}")
    finally:
        print("Tracker context exited.")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    if args.source == "uff":
        if not args.uff:
            print("Specify --uff when using --source=uff", file=sys.stderr)
            raise SystemExit(2)
        if not Path(args.uff).exists():
            print(f"UFF file not found: {args.uff}", file=sys.stderr)
            raise SystemExit(2)
    run_holoscan_app(args)


if __name__ == "__main__":
    main()
